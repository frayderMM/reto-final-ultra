#!/usr/bin/env python3
"""Nodo de decision de intersecciones y maquina de estados principal.

Es el UNICO nodo que escribe en ``/cmd_vel``: centraliza toda decision
de movimiento para evitar que dos publicadores manden comandos
contradictorios al mismo tiempo. Mientras el estado es
AVANZAR_PARALELO, reenvia la sugerencia de ``wall_follower_node``
(``/wall_follow/cmd_vel_suggestion``); en el resto de estados calcula
sus propios comandos (girar detenido-lento, alinear, detener).

Maquina de estados (ver logica_pared_derecha_robot.md y
DETALLE RETO 3.md):

    INICIAR -> AVANZAR_PARALELO -> DETECTAR_CRUCE -> BUSCAR_PARE
    -> DECIDIR -> PAUSA_GIRO -> GIRAR -> ALINEAR -> VERIFICAR_META
    -> (META o vuelve a AVANZAR_PARALELO)

    PAUSA_GIRO (fuera de la lista original del documento de referencia)
    es una espera fija de ``tiempo_pausa_antes_girar_s`` con el robot
    detenido entre "ya decidi" y "empiezo a girar", para que el giro se
    vea como un movimiento separado del avance.

    En logica_dos_reglas, cada ``distancia_chequeo_pared_m`` de avance
    en linea recta se pasa por ``PAUSA_CHEQUEO_PARED`` en vez de
    PAUSA_GIRO: detenido ``tiempo_chequeo_pared_s`` (0.5s) y verifica con
    distancia PUNTUAL (no el ajuste de linea) si el lado derecho esta
    ocupado (pared) o vacio antes de comprometerse a girar -- ver
    ``_handle_pausa_chequeo_pared``.

Se agrega un estado adicional ``DETENIDO`` (fuera de la lista pedida)
solo como red de seguridad ante un limite de celdas recorridas sin
llegar a la meta (evita loops infinitos por fallas de sensor); no
reemplaza ni altera el flujo principal solicitado.

Nota sobre giros con chasis Ackermann: un vehiculo con direccion
Ackermann no puede rotar sobre su propio eje (radio de giro cero). El
estado GIRAR aproxima el "giro detenido" del documento de referencia
con un arco de avance lento y radio de giro pequeno (velocidad lineal
baja + angular maxima), usando el yaw de ``/odom_raw`` como
referencia de cierre en vez de tiempo fijo. Esto se debe calibrar en
pista (ver README).
"""

import json
import math
from types import SimpleNamespace

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, String

from capytown_g0_granprix import motion_events as EV
from capytown_g0_granprix.motion_geometry import angle_diff, normalize_angle, yaw_from_quaternion
from capytown_g0_granprix.motion_grid import GridTracker
from capytown_g0_granprix.motion_ruta import (
    RouteExplorer, bfs_ruta, celdas_a_movimientos)
from capytown_g0_granprix.motion_lidar import (
    ZoneWindow, compute_robot_frame_angles, compute_zone_distance, fit_wall_line)


class MazeSolverNode(Node):

    def __init__(self):
        super().__init__('maze_solver')
        self._declare_parameters()
        self._read_parameters()

        self._grid = GridTracker.from_cell_name(self._celda_inicio, self._heading_inicial)

        # Estado de la maquina
        self._state = 'INICIAR'
        self._terminado = False

        # Datos de sensores (ultimo valor recibido)
        self._zones = None
        self._zones_ready = False
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._yaw = 0.0
        self._odom_ready = False
        self._pare_activo = False
        self._pare_anterior = False
        self._pare_pendiente = False
        self._pare_ignorar_xy = None
        self._freno_pare_start = None
        self._wall_follow_cmd = Twist()

        # Variables de trabajo por estado
        self._cell_start_xy = (0.0, 0.0)
        self._avance_chequeo_start_xy = (0.0, 0.0)
        self._num_celdas = 0
        self._cruce_muestras = None
        self._derecha_libre = False
        self._frente_libre = False
        self._izquierda_libre = False
        self._buscar_pare_start = None
        self._pare_hold_start = None
        self._celdas_pare_respetadas = set()
        self._decision_actual = 'NINGUNO'
        self._giro_objetivo = 0.0
        self._alinear_start = None
        self._pausa_giro_start = None

        self._esperando_obstaculo = False
        self._espera_obstaculo_inicio = None
        self._retrocediendo_obstaculo = False
        self._retroceso_obstaculo_xy0 = (0.0, 0.0)
        self._contador_frente_dos_reglas = 0
        self._yaw_inicio_giro = 0.0
        self._imu_acum_giro = 0.0
        self._imu_t_prev = None
        self._pausa_chequeo_start = None
        self._contador_derecha_libre = 0
        self._chequeo_por_frente = False
        self._giro_vacio_fase = 0
        self._giro_vacio_repeticiones = 0
        self._avance_fijo_inicio_xy = (0.0, 0.0)

        # --- FASE 2 (ruta corta / speed-run), todo aditivo ---
        self._explorer = (
            RouteExplorer.from_cell_name(
                self._ruta_celda_inicio, self._ruta_heading_inicial)
            if self._ruta_activa else None)
        self._ruta_start_cell = self._explorer.cell if self._explorer else None
        self._verde_anterior = False
        self._meta_cell = None
        self._ruta_last_xy = None            # ancla para contar avances de celda
        self._ruta_dist_acc = 0.0            # distancia recta acumulada sin celda
        self._ruta_celdas = None             # ruta BFS [(c,r),...]
        self._ruta_movimientos = None        # guion [{'giro','celdas'}, ...]
        self._ruta_idx = 0
        self._ruta_fase = 'GIRO'             # GIRO|AVANCE dentro de SEGUIR_RUTA
        self._ruta_giro_yaw0 = 0.0
        self._ruta_giro_restante = None      # sub-giros de 90 pendientes (ATRAS=2)
        self._ruta_avance_xy0 = (0.0, 0.0)
        self._ruta_json = None
        self._ruta_pub_counter = 0

        # Monitor anti-vuelta-completa (>360 grados girando)
        self._yaw_prev_360 = None
        self._rot_acum = 0.0
        self._correccion_signo = 1.0
        self._yaw_correccion0 = 0.0

        # --- Metricas Gran Prix (para metrics_logger.py via /maze/metricas) ---
        # metrics_logger.py escucha este mismo topico esperando estos campos
        # ademas de los de telemetria de visualizador_web -- se agregan aca
        # sin tocar la logica de manejo. Se congelan en un snapshot al
        # llegar a META (ver _on_verde) para que tiempo_s/long_ruta_cm
        # reflejen el trayecto INICIO->META y no seguir sumando si el
        # carrito continua dando la vuelta al laberinto despues de verla.
        self._tiempo_inicio = None
        self._distancia_total_m = 0.0
        self._odom_prev_xy = None
        self._contador_colisiones = 0
        self._contador_pare_detectados = 0
        self._contador_pare_respetados = 0
        self._metricas_meta = None
        # Contador de giros fisicos de 90 reales -- ver nota junto a
        # 'giros_fisicos' en _publish_twist.
        self._contador_giros_fisicos = 0

        self._STATE_HANDLERS = {
            'INICIAR': self._handle_iniciar,
            'AVANZAR_PARALELO': self._handle_avanzar_paralelo,
            'DETECTAR_CRUCE': self._handle_detectar_cruce,
            'BUSCAR_PARE': self._handle_buscar_pare,
            'DECIDIR': self._handle_decidir,
            'PAUSA_GIRO': self._handle_pausa_giro,
            'PAUSA_CHEQUEO_PARED': self._handle_pausa_chequeo_pared,
            'GIRAR': self._handle_girar,
            'AVANCE_GIRO_VACIO': self._handle_avance_giro_vacio,
            'ALINEAR': self._handle_alinear,
            'VERIFICAR_META': self._handle_verificar_meta,
            'ESPERA_RUTA': self._handle_espera_ruta,
            'SEGUIR_RUTA': self._handle_seguir_ruta,
            'CORREGIR_GIRO': self._handle_corregir_giro,
            'META': self._handle_meta,
            'DETENIDO': self._handle_detenido,
        }

        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)
        self._event_pub = self.create_publisher(String, self._event_topic, 10)
        self._state_pub = self.create_publisher(String, self._robot_state_topic, 10)
        self._metrics_pub = self.create_publisher(String, '/maze/metricas', 10)
        self._ruta_pub = self.create_publisher(String, self._ruta_topic, 10)

        self.create_subscription(
            LaserScan, self._scan_topic, self._on_scan, QoSPresetProfiles.SENSOR_DATA.value
        )
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, 10)
        self.create_subscription(
            Imu, self._imu_topic, self._on_imu, QoSPresetProfiles.SENSOR_DATA.value
        )
        self.create_subscription(Bool, self._pare_topic, self._on_pare, 10)
        # El verde no controla el mapeo: solo marca la celda de la meta para
        # poder calcular la ruta corta al regresar a la base (fase 2).
        self.create_subscription(Bool, self._verde_topic, self._on_verde, 10)
        # Dos comandos de la ruta corta (fase 2): calcular (no mueve) y partir.
        self.create_subscription(
            Bool, self._calcular_ruta_topic, self._on_calcular_ruta, 10)
        self.create_subscription(
            Bool, self._iniciar_ruta_topic, self._on_iniciar_ruta, 10)

        self.create_timer(1.0 / self._control_rate_hz, self._on_timer)

        self.get_logger().info(
            f'maze_solver referencia listo: inicio={self._celda_inicio} meta={self._celda_meta} '
            f'heading_inicial={self._heading_inicial}'
        )

    # ------------------------------------------------------------------
    # Parametros
    # ------------------------------------------------------------------
    def _declare_parameters(self):
        defaults = {
            'scan_topic': '/scan',
            'front_offset_deg': 180.0,
            'invert_left_right': False,
            'max_range_use_m': 4.0,
            'front_window_deg': [-15.0, 15.0],
            'front_narrow_window_deg': [-5.0, 5.0],
            'right_front_window_deg': [-75.0, -45.0],
            'right_window_deg': [-110.0, -70.0],
            'right_rear_window_deg': [-135.0, -105.0],
            'left_window_deg': [70.0, 110.0],
            'right_side_window_deg': [-110.0, -70.0],
            'left_side_window_deg': [70.0, 110.0],
            'min_puntos_linea': 6,
            'right_wall_max_range_m': 0.50,
            'left_wall_max_range_m': 0.50,
            'outlier_max_iter': 3,
            'outlier_residuo_m': 0.03,
            'lidar_zones_topic': '/lidar_zones',
            'odom_topic': '/odom_raw',
            # Deteccion de patinaje en giros: compara cuanto giro segun la
            # odometria de rueda vs. cuanto giro segun el giroscopio del IMU
            # (no depende de las ruedas). Una diferencia grande indica que
            # una rueda patino durante el giro.
            'imu_topic': '/imu',
            'umbral_patinaje_deg': 8.0,
            'cmd_vel_topic': '/cmd_vel',
            'wall_follow_topic': '/wall_follow/cmd_vel_suggestion',
            'pare_topic': '/pare_detectado',
            'event_topic': '/robot_event',
            'robot_state_topic': '/maze/estado',
            'usar_camara': True,
            'control_rate_hz': 20.0,
            # Modo de prueba: si es true, se saltan DETECTAR_CRUCE y
            # BUSCAR_PARE -- decide con una lectura unica (sin
            # confirmar con varias muestras). ALINEAR SI corre (no se
            # salta): es el paso que corrige el giro contra la pared
            # real via LiDAR en vez de confiar solo en el angulo
            # objetivo fijo + odometria. Util para calibrar el giro de
            # forma aislada, con el feedback de alineacion incluido.
            'modo_simplificado': True,
            # SOLO DOS REGLAS (rama logica-alternativa, para probar en
            # hardware lo mismo que sim_local/run_sim_laberinto.py::
            # _correr_logica_simple): avanzar recto mientras el frente
            # este libre; si hay obstaculo al frente, girar 90 grados a
            # la IZQUIERDA fijo (sin mirar derecha/izquierda, sin
            # seguir pared, sin celda, sin ALINEAR). Cuando esta en
            # true, IGNORA modo_simplificado y el resto de la maquina
            # de estados de cruce/PARE -- dejar en false para la
            # corrida real de competencia.
            'logica_dos_reglas': True,
            # Que lado sigue logica_dos_reglas: true = pared IZQUIERDA
            # (left_line_*, left/left_valid), false = pared DERECHA
            # (right_line_*, right/right_valid, el original). Al
            # cambiar de lado tambien se espejan las direcciones de
            # giro: obstaculo al frente gira hacia el lado NO seguido
            # (aleja de la pared que sigue) y "vacio" gira hacia el
            # lado SEGUIDO (entra al hueco que aparecio ahi). Ver
            # _lado_seguido_* y _direccion_* en el codigo.
            'seguir_pared_izquierda': True,
            'velocidad_recta_mps': 0.15,
            # Correccion lateral de logica_dos_reglas: usa el AJUSTE DE
            # LINEA (right_line_*, angulo + distancia) en vez de la
            # distancia puntual right_valid/right -- evita confundir una
            # pared vista en diagonal con un obstaculo nuevo, porque el
            # ajuste de linea da el angulo real de la pared en vez de un
            # numero suelto. Misma formula que wall_follow_control.
            # calcular_comando, re-declarada aqui porque este modo NO
            # usa wall_follow_cmd en absoluto.
            'distancia_objetivo_m': 0.12,
            # Correccion lateral mas estricta: mas ganancia en el termino
            # de distancia (el que corrige que tan pegado va a la pared
            # seguida) y algo mas en angulo; angular_max sube junto para
            # que la correccion mas fuerte no se sature de inmediato.
            'ganancia_angulo_recta': 2.0,
            'ganancia_distancia_recta': 2.0,
            'angular_max_recta_radps': 0.6,
            # Confirmacion de N ciclos seguidos con front_narrow
            # bloqueado antes de girar -- un solo vistazo diagonal de un
            # ciclo (100% ruido/transitorio) no alcanza para disparar un
            # giro, tiene que sostenerse.
            'frente_confirmaciones_ciclos': 3,
            # Chequeo PERIODICO del lado seguido (seguir_pared_
            # izquierda), no deteccion continua por LiDAR (en la
            # practica no distinguia bien "pared" de "hueco" -- ver
            # commit anterior): cada distancia_chequeo_pared_m de
            # avance en linea recta, se detiene por completo
            # (PAUSA_CHEQUEO_PARED) tiempo_chequeo_pared_s y verifica
            # con distancia PUNTUAL (no la linea) si el lado seguido
            # esta ocupado o vacio.
            'distancia_chequeo_pared_m': 0.12,
            'tiempo_chequeo_pared_s': 0.5,
            # "Lado derecho vacio" tiene que sostenerse esta cantidad
            # de ciclos SEGUIDOS (no una sola lectura) antes de
            # comprometerse a girar -- un giro a la derecha es una
            # decision cara de revertir. "Ocupado" no necesita esto.
            'chequeo_pared_confirmaciones_ciclos': 5,
            # Al girar por "lado seguido vacio" (no por obstaculo al
            # frente): giro de 90 (mismo mecanismo de giro fijo,
            # _handle_girar_dinamico) + avance recto de
            # avance_giro_vacio_m -- y RECIEN AHI verifica de nuevo si
            # el lado seguido sigue vacio. Si sigue vacio, repite el
            # par giro+avance; si ya encuentra pared, retoma
            # AVANZAR_PARALELO. giro_vacio_max_repeticiones es el tope
            # de seguridad (evita girar para siempre en un espacio muy
            # abierto). Ver AVANCE_GIRO_VACIO/_handle_avance_giro_vacio.
            'avance_giro_vacio_m': 0.12,
            'giro_vacio_max_repeticiones': 2,
            # Giro de logica_dos_reglas: cierra el lazo por odometria
            # contra angulo_giro_deg (90, fijo y exacto -- ver
            # _handle_girar_dinamico). angulo_maximo_giro_deg es tope
            # de seguridad adicional (por si el odometro se traba).
            'angulo_maximo_giro_deg': 150.0,
            'umbral_frente_pared_m': 0.30,
            'umbral_frente_libre_m': 0.35,
            'umbral_lado_libre_m': 0.30,
            # Regla general de seguridad (siempre activa, en cualquier
            # estado): objeto al frente mas cerca que esto -> detenerse
            # de inmediato, esperar y volver a preguntar si esta libre. Se
            # subio de 0.10 a 0.15 para frenar antes y evitar choques (el
            # chasis Ackermann tiene inercia y el cono ancho promedia).
            'umbral_colision_m': 0.15,
            # Al bloquearse por obstaculo frontal, retrocede esta distancia
            # (odometria) antes de quedar esperando -- despega del objeto en
            # vez de solo detenerse en seco. El retroceso gira hacia la
            # IZQUIERDA mientras retrocede (arco, no en linea recta).
            'retroceso_obstaculo_m': 0.10,
            'velocidad_retroceso_obstaculo_mps': 0.06,
            'velocidad_retroceso_obstaculo_angular_radps': 0.9,
            # Distancia lateral MINIMA al lado seguido (izquierda): si el LiDAR
            # ve la pared izquierda mas cerca que esto, se corrige alejandose
            # con fuerza para no rozarla (anti-choque lateral, ver
            # _handle_avanzar_paralelo_dos_reglas).
            'umbral_lateral_min_m': 0.07,
            # Correccion anti-vuelta-completa: si el robot acumula MAS de 360
            # grados girando (se puso a dar vueltas), hace un giro de
            # correccion_giro_grados al lado CONTRARIO para reorientarse y
            # retoma el avance. Evita que quede girando en redondo.
            'correccion_giro_360': True,
            'correccion_giro_grados': 10.0,
            'tiempo_espera_obstaculo_s': 2.0,
            'distancia_celda_m': 5.00,
            'margen_avance_m': 0.05,
            'muestras_confirmacion': 5,
            'consenso_minimo': 4,
            'velocidad_giro_lineal_mps': 0.06,
            'velocidad_giro_angular_radps': 0.6,
            'tolerancia_giro_deg': 4.0,
            # Angulo objetivo de giro para DERECHA/IZQUIERDA (ATRAS
            # siempre es 180, no usa este valor). 90 es el giro "real"
            # de una esquina en grilla; un poco mas (ej. 95) compensa
            # que el arco Ackermann suele quedar corto del objetivo.
            'angulo_giro_deg': 90.0,
            # Pausa fija (segundos) con el robot detenido entre DECIDIR
            # (ya sabe que va a girar) y el inicio del arco de GIRAR --
            # pedido para que el giro sea un movimiento claramente
            # separado del avance, no una transicion instantanea.
            'tiempo_pausa_antes_girar_s': 1.0,
            'tolerancia_alineacion_m': 0.02,
            'tiempo_max_alinear_s': 4.0,
            'velocidad_alineacion_lineal_mps': 0.06,
            'velocidad_alineacion_angular_radps': 0.3,
            # FRENO_PARE solo congela el movimiento; al terminar retoma
            # exactamente el estado interno que estaba ejecutando.
            'tiempo_pare_s': 5.0,
            # Tras cumplirse tiempo_pare_s (terminado el FRENO_PARE), ignora
            # nuevas detecciones de rojo hasta que el robot avance esta
            # distancia (por odometria) desde el punto donde reanudo. Evita
            # que el mismo cartel (u otro reflejo cercano) dispare un
            # segundo FRENO_PARE sin haberse alejado de la senal.
            'distancia_ignorar_pare_m': 0.60,
            'tiempo_espera_camara_s': 0.5,
            'celda_inicio': 'A4',
            'celda_meta': 'F1',
            'heading_inicial': 'NORTE',
            'max_celdas_recorridas': 60,
            # Factores de correccion de escala del odometro (calibrados en
            # pista: avance real 76 cm / odometro 78.3 cm y giro real 90 /
            # odometro 90.92). Dejar en 1.0 si se recalibra desde cero.
            'factor_dist_odom': 0.9474,
            'factor_ang_odom': 0.9899,
            # ----------------------------------------------------------------
            # FASE 2 (speed-run): tras mapear siguiendo la pared izquierda y
            # ver la meta (verde), se calcula la ruta mas corta con BFS sobre
            # las celdas exploradas y se MANEJA obedeciendo solo la ruta + la
            # anticolision (sin logica de que lado seguir). El carrito NUNCA
            # parte solo: hay dos comandos (calcular y partir, abajo). Todo es
            # ADITIVO: no altera ninguna decision del mapeo. Con
            # ruta_activa=false el comportamiento es identico al mapeo puro.
            'ruta_activa': True,
            'verde_topic': '/verde_detectado',
            'ruta_topic': '/maze/ruta_corta',
            # DOS comandos para la ruta corta (el carrito NUNCA parte solo):
            # - calcular_ruta_topic: calcula la ruta y la pinta de amarillo,
            #   dejando el carrito DETENIDO (no se mueve).
            # - iniciar_ruta_topic: recien con este comando el carrito PARTE y
            #   maneja la ruta.
            'calcular_ruta_topic': '/maze/calcular_ruta',
            'iniciar_ruta_topic': '/maze/iniciar_ruta',
            # Distancia por celda al MANEJAR la ruta (y para discretizar los
            # avances del mapeo en celdas). Debe coincidir con CELDA_M del
            # visualizador (0.30) para que el amarillo cuadre con el mapa.
            'tamano_celda_m': 0.30,
            # Celda de arranque en la grilla 12x8 del mapa web (A7 = celda de
            # inicio del carrito; coincide con INICIO_POS del visualizador).
            'ruta_celda_inicio': 'A7',
            'ruta_heading_inicial': 'NORTE',
            # El punto de partida es SIEMPRE el mismo (inicio). Con esto en True,
            # el guion de manejo se arma asumiendo que el carrito estara en el
            # INICIO mirando ruta_heading_inicial (NORTE) -- pensado para que lo
            # coloques ahi antes de dar el comando de partir. En False usa el
            # rumbo interno de odometria (solo si NO recolocas el carrito).
            'ruta_asume_rumbo_inicial': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self):
        g = lambda name: self.get_parameter(name).value  # noqa: E731

        self._scan_topic = g('scan_topic')
        self._front_offset_rad = math.radians(float(g('front_offset_deg')))
        self._lidar_sign = -1 if bool(g('invert_left_right')) else 1
        self._max_range_use = float(g('max_range_use_m'))
        self._lidar_windows = {
            'front': ZoneWindow(*g('front_window_deg')),
            'front_narrow': ZoneWindow(*g('front_narrow_window_deg')),
            'right_front': ZoneWindow(*g('right_front_window_deg')),
            'right': ZoneWindow(*g('right_window_deg')),
            'right_rear': ZoneWindow(*g('right_rear_window_deg')),
            'left': ZoneWindow(*g('left_window_deg')),
        }
        self._right_side_window = ZoneWindow(*g('right_side_window_deg'))
        self._left_side_window = ZoneWindow(*g('left_side_window_deg'))
        self._min_puntos_linea = int(g('min_puntos_linea'))
        self._right_wall_max_range = float(g('right_wall_max_range_m'))
        self._left_wall_max_range = float(g('left_wall_max_range_m'))
        self._outlier_max_iter = int(g('outlier_max_iter'))
        self._outlier_residuo = float(g('outlier_residuo_m'))

        self._lidar_zones_topic = g('lidar_zones_topic')
        self._odom_topic = g('odom_topic')
        self._imu_topic = g('imu_topic')
        self._umbral_patinaje = float(g('umbral_patinaje_deg'))
        self._cmd_vel_topic = g('cmd_vel_topic')
        self._wall_follow_topic = g('wall_follow_topic')
        self._pare_topic = g('pare_topic')
        self._event_topic = g('event_topic')
        self._robot_state_topic = g('robot_state_topic')

        self._usar_camara = bool(g('usar_camara'))
        self._control_rate_hz = float(g('control_rate_hz'))
        self._modo_simplificado = bool(g('modo_simplificado'))
        self._logica_dos_reglas = bool(g('logica_dos_reglas'))
        self._seguir_izquierda = bool(g('seguir_pared_izquierda'))
        self._velocidad_recta = float(g('velocidad_recta_mps'))
        self._distancia_objetivo_recta = float(g('distancia_objetivo_m'))
        self._ganancia_angulo_recta = float(g('ganancia_angulo_recta'))
        self._ganancia_distancia_recta = float(g('ganancia_distancia_recta'))
        self._angular_max_recta = float(g('angular_max_recta_radps'))
        self._frente_confirmaciones_ciclos = int(g('frente_confirmaciones_ciclos'))
        self._distancia_chequeo_pared = float(g('distancia_chequeo_pared_m'))
        self._chequeo_pared_confirmaciones_ciclos = int(g('chequeo_pared_confirmaciones_ciclos'))
        self._avance_giro_vacio = float(g('avance_giro_vacio_m'))
        self._giro_vacio_max_repeticiones = int(g('giro_vacio_max_repeticiones'))
        self._tiempo_chequeo_pared = float(g('tiempo_chequeo_pared_s'))
        self._contador_frente_dos_reglas = 0
        self._angulo_maximo_giro_rad = math.radians(float(g('angulo_maximo_giro_deg')))

        self._umbral_frente_pared = float(g('umbral_frente_pared_m'))
        self._umbral_frente_libre = float(g('umbral_frente_libre_m'))
        self._umbral_lado_libre = float(g('umbral_lado_libre_m'))
        self._umbral_colision = float(g('umbral_colision_m'))
        self._retroceso_obstaculo = float(g('retroceso_obstaculo_m'))
        self._v_retroceso_obstaculo = float(g('velocidad_retroceso_obstaculo_mps'))
        self._w_retroceso_obstaculo = float(g('velocidad_retroceso_obstaculo_angular_radps'))
        self._umbral_lateral_min = float(g('umbral_lateral_min_m'))
        self._correccion_giro_360 = bool(g('correccion_giro_360'))
        self._correccion_giro_rad = math.radians(float(g('correccion_giro_grados')))
        self._distancia_celda = float(g('distancia_celda_m'))
        self._margen_avance = float(g('margen_avance_m'))

        self._muestras_confirmacion = int(g('muestras_confirmacion'))
        self._consenso_minimo = int(g('consenso_minimo'))

        self._v_giro_lineal = float(g('velocidad_giro_lineal_mps'))
        self._v_giro_angular = float(g('velocidad_giro_angular_radps'))
        self._tolerancia_giro_rad = math.radians(float(g('tolerancia_giro_deg')))
        self._angulo_giro_rad = math.radians(float(g('angulo_giro_deg')))
        self._tiempo_pausa_antes_girar = float(g('tiempo_pausa_antes_girar_s'))

        self._tolerancia_alineacion = float(g('tolerancia_alineacion_m'))
        self._tiempo_max_alinear = float(g('tiempo_max_alinear_s'))
        self._v_alinear_lineal = float(g('velocidad_alineacion_lineal_mps'))
        self._v_alinear_angular = float(g('velocidad_alineacion_angular_radps'))

        self._tiempo_pare = float(g('tiempo_pare_s'))
        self._distancia_ignorar_pare = float(g('distancia_ignorar_pare_m'))
        self._tiempo_espera_camara = float(g('tiempo_espera_camara_s'))

        self._tiempo_espera_obstaculo = float(g('tiempo_espera_obstaculo_s'))

        self._celda_inicio = str(g('celda_inicio'))
        self._celda_meta = str(g('celda_meta'))
        self._heading_inicial = str(g('heading_inicial'))
        self._max_celdas = int(g('max_celdas_recorridas'))

        self._factor_dist_odom = float(g('factor_dist_odom'))
        self._factor_ang_odom = float(g('factor_ang_odom'))

        self._ruta_activa = bool(g('ruta_activa'))
        self._verde_topic = g('verde_topic')
        self._ruta_topic = g('ruta_topic')
        self._calcular_ruta_topic = g('calcular_ruta_topic')
        self._iniciar_ruta_topic = g('iniciar_ruta_topic')
        self._tamano_celda = float(g('tamano_celda_m'))
        self._ruta_celda_inicio = str(g('ruta_celda_inicio'))
        self._ruta_heading_inicial = str(g('ruta_heading_inicial'))
        self._ruta_asume_rumbo_inicial = bool(g('ruta_asume_rumbo_inicial'))

    # ------------------------------------------------------------------
    # Callbacks de suscripcion
    # ------------------------------------------------------------------
    def _on_scan(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=float)
        angles = compute_robot_frame_angles(
            ranges, msg.angle_min, msg.angle_increment,
            self._front_offset_rad, self._lidar_sign)
        max_use = min(float(msg.range_max), self._max_range_use)
        z = SimpleNamespace()
        for name in ('front', 'front_narrow', 'right_front', 'right', 'right_rear', 'left'):
            distance, valid = compute_zone_distance(
                ranges, angles, msg.range_min, max_use, self._lidar_windows[name])
            setattr(z, name, distance)
            setattr(z, f'{name}_valid', valid)
        for side, window, max_wall in (
                ('right', self._right_side_window, self._right_wall_max_range),
                ('left', self._left_side_window, self._left_wall_max_range)):
            angle, distance, valid = fit_wall_line(
                ranges, angles, msg.range_min, min(max_use, max_wall), window,
                self._min_puntos_linea, self._outlier_max_iter, self._outlier_residuo)
            setattr(z, f'{side}_line_angle_rad', angle)
            setattr(z, f'{side}_line_distance_m', distance)
            setattr(z, f'{side}_line_valid', valid)
        self._zones = z
        self._zones_ready = True

    def _on_odom(self, msg: Odometry):
        # Correccion de escala del odometro (medida en pista, ver README):
        # el ROSMASTER R2 sobreestima tanto distancia como angulo girado,
        # de forma consistente, por lo que se corrige con un factor fijo.
        self._odom_x = msg.pose.pose.position.x * self._factor_dist_odom
        self._odom_y = msg.pose.pose.position.y * self._factor_dist_odom
        self._yaw = yaw_from_quaternion(msg.pose.pose.orientation) * self._factor_ang_odom
        self._odom_ready = True

        # long_ruta_cm (metrica Gran Prix): odometro acumulado sin importar
        # el estado -- mismo patron que visualizador_web.odom_cm.
        if self._odom_prev_xy is not None:
            dx = self._odom_x - self._odom_prev_xy[0]
            dy = self._odom_y - self._odom_prev_xy[1]
            self._distancia_total_m += math.hypot(dx, dy)
        self._odom_prev_xy = (self._odom_x, self._odom_y)

    def _on_imu(self, msg: Imu):
        """Integra el giroscopio (angular_velocity.z) SOLO durante GIRAR,
        para comparar contra el angulo girado por odometria de rueda al
        terminar el giro y detectar patinaje (ver _handle_girar)."""
        ahora = self.get_clock().now()
        if self._state == 'GIRAR' and self._imu_t_prev is not None:
            dt = (ahora - self._imu_t_prev).nanoseconds / 1e9
            self._imu_acum_giro += msg.angular_velocity.z * dt
        self._imu_t_prev = ahora

    def _on_pare(self, msg: Bool):
        detectado = bool(msg.data)
        en_cooldown = False
        if self._pare_ignorar_xy is not None:
            dx = self._odom_x - self._pare_ignorar_xy[0]
            dy = self._odom_y - self._pare_ignorar_xy[1]
            en_cooldown = math.hypot(dx, dy) < self._distancia_ignorar_pare
        # Solo el flanco de entrada dispara el freno. Mantener el cartel
        # delante de la cámara no reinicia continuamente los cinco segundos.
        # Ademas, tras cumplirse el FRENO_PARE (ver _on_timer) se ignoran
        # nuevos flancos hasta avanzar distancia_ignorar_pare_m: evita que
        # el mismo cartel dispare un segundo FRENO_PARE sin haberse alejado.
        if detectado and not self._pare_anterior:
            if en_cooldown:
                # Diagnostico: confirma que el cooldown esta activo y
                # descartando flancos repetidos del mismo cartel.
                dx = self._odom_x - self._pare_ignorar_xy[0]
                dy = self._odom_y - self._pare_ignorar_xy[1]
                self._publish_event(
                    EV.PARE_FALSO,
                    f'rojo ignorado (cooldown): avanzo {math.hypot(dx, dy):.2f}m '
                    f'de {self._distancia_ignorar_pare:.2f}m requeridos'
                )
            else:
                self._pare_pendiente = True
                self._contador_pare_detectados += 1
        self._pare_activo = detectado
        self._pare_anterior = detectado

    def _on_wall_follow(self, msg: Twist):
        self._wall_follow_cmd = msg

    def _on_verde(self, msg: Bool):
        """Marca la celda de la meta la primera vez que se confirma el verde.

        No cambia el estado ni el comando: el verde sigue sin afectar el
        mapeo. Solo registra donde esta la meta para la ruta corta (fase 2).
        """
        activo = bool(msg.data)
        if (activo and not self._verde_anterior and self._ruta_activa
                and self._explorer is not None and self._meta_cell is None):
            self._meta_cell = self._explorer.cell
            # Congela tiempo_s/long_ruta_cm aca (INICIO->META): si el
            # carrito sigue dando la vuelta al laberinto despues de ver el
            # verde (logica_dos_reglas no se detiene sola), esos dos campos
            # no deben seguir creciendo con ese recorrido extra.
            self._metricas_meta = self._metricas_actuales()
            self._publish_event(
                EV.META, f'meta (verde) registrada en celda {self._meta_cell}'
            )
        self._verde_anterior = activo

    def _on_calcular_ruta(self, msg: Bool):
        """Comando 1: FRENAR el mapeo + CALCULAR la ruta.

        Frena y calcula la ruta SOLO si ya se detecto el verde (meta); ahi
        detiene el carrito (sin matar el nodo: conserva el grafo) y lo deja
        quieto en ESPERA_RUTA con el amarillo pintado. Si aun no vio el verde,
        NO frena: sigue mapeando. Es la unica forma de frenar -- NO uses Ctrl+C,
        que borraria el mapeo. Nunca parte solo."""
        if not bool(msg.data):
            return
        if not self._ruta_activa or self._explorer is None:
            self._publish_event(EV.TIMEOUT, 'calcular_ruta ignorado: ruta_activa=false')
            return
        if self._terminado or self._state in ('ESPERA_RUTA', 'SEGUIR_RUTA', 'META'):
            return
        if not self._calcular_ruta():
            return   # sin verde/meta todavia: NO frena, sigue mapeando
        self._publish_twist(Twist())
        self._publish_event(
            EV.META,
            'DETENIDO + ruta calculada (amarillo). '
            'Envia /maze/iniciar_ruta para PARTIR.'
        )
        self._set_state('ESPERA_RUTA')

    def _on_iniciar_ruta(self, msg: Bool):
        """Comando 2: PARTIR. Recien con este comando el carrito arranca y
        maneja la ruta corta. Se puede REPETIR cuantas veces quieras: al
        terminar la ruta el carrito vuelve a ESPERA_RUTA (detenido) y este
        comando la maneja de nuevo desde el inicio (colocalo ahi). Nunca parte
        solo. Requiere que la ruta ya este calculada (/maze/calcular_ruta)."""
        if not bool(msg.data):
            return
        if not self._ruta_activa or self._explorer is None:
            self._publish_event(EV.TIMEOUT, 'iniciar_ruta ignorado: ruta_activa=false')
            return
        if self._terminado or self._state == 'SEGUIR_RUTA':
            return   # ya manejando (o terminado): no reiniciar a mitad
        if self._ruta_movimientos is None and not self._calcular_ruta():
            self._publish_event(
                EV.TIMEOUT,
                'iniciar_ruta ignorado: no hay ruta calculada '
                '(usa /maze/calcular_ruta; falta ver el verde?)'
            )
            return
        self._ruta_idx = 0
        self._ruta_fase = 'GIRO'
        self._ruta_giro_restante = None
        self._publish_event(EV.INICIO, 'PARTIR: manejando la ruta corta')
        self._set_state('SEGUIR_RUTA')

    # ------------------------------------------------------------------
    # Ciclo de control principal
    # ------------------------------------------------------------------
    def _on_timer(self):
        if not (self._odom_ready and self._zones_ready):
            return

        # FRENO_PARE es una capa de pausa: NO cambia self._state ni ejecuta
        # decisiones. Solo manda velocidad cero y congela los relojes internos.
        if self._freno_pare_start is not None:
            elapsed = (self.get_clock().now() - self._freno_pare_start).nanoseconds / 1e9
            if elapsed < self._tiempo_pare:
                self._state_pub.publish(String(data='FRENO_PARE'))
                self._publish_twist(Twist())
                return

            self._congelar_relojes_durante(elapsed)
            self._freno_pare_start = None
            # Ignora nuevos flancos de rojo (incluye los que hayan quedado
            # pendientes por parpadeo durante la espera) hasta que el robot
            # avance distancia_ignorar_pare_m desde este punto -- recien
            # cumplido el tiempo_pare_s de espera.
            self._pare_pendiente = False
            self._pare_ignorar_xy = (self._odom_x, self._odom_y)
            self._celdas_pare_respetadas.add(self._grid.cell)
            self._contador_pare_respetados += 1
            self._publish_event(
                EV.PARE_RESPETADO,
                f'PARE respetado {elapsed:.1f}s; continúa en {self._state}',
            )
            self._state_pub.publish(String(data=self._state))
        elif self._pare_pendiente and self._state not in ('META', 'DETENIDO'):
            self._pare_pendiente = False
            self._freno_pare_start = self.get_clock().now()
            self._publish_event(
                EV.PARE_DETECTADO,
                f'rojo detectado: FRENO_PARE {self._tiempo_pare:.1f}s '
                f'sin abandonar {self._state}',
            )
            self._state_pub.publish(String(data='FRENO_PARE'))
            self._publish_twist(Twist())
            return
        elif self._pare_pendiente:
            # En META/DETENIDO el carrito ya está parado.
            self._pare_pendiente = False

        if self._handle_obstaculo_frente():
            return

        self._STATE_HANDLERS[self._state]()

        # Grabacion pasiva del grafo de celdas para la ruta corta (fase 2).
        # Corre DESPUES del handler y no modifica ninguna decision de mapeo.
        self._registrar_avance_ruta()

        # Monitor: si acumula mas de 360 grados girando, corrige al contrario.
        self._monitor_rotacion()

    def _congelar_relojes_durante(self, segundos: float):
        """Impide que FRENO_PARE consuma tiempos de la maniobra pausada."""
        pausa = Duration(nanoseconds=int(segundos * 1e9))
        for nombre in (
                '_buscar_pare_start', '_pare_hold_start', '_alinear_start',
                '_pausa_giro_start', '_espera_obstaculo_inicio',
                '_pausa_chequeo_start'):
            instante = getattr(self, nombre, None)
            if instante is not None:
                setattr(self, nombre, instante + pausa)

    def _handle_obstaculo_frente(self) -> bool:
        """Regla general de seguridad, activa en cualquier estado.

        Si hay un objeto al frente mas cerca que ``umbral_colision_m``,
        retrocede ``retroceso_obstaculo_m`` para despegarse de el y luego
        espera ``tiempo_espera_obstaculo_s``, comprobando si ya esta libre;
        si sigue bloqueado, reinicia la espera (queda preguntando en bucle
        hasta que se libere). Retorna True si este ciclo ya publico un
        comando (el llamador debe omitir el despacho normal de estados).
        """
        if self._terminado:
            return False

        z = self._zones
        # Anti-choque con LiDAR: bloquea si el cono ANCHO o el ANGOSTO ven algo
        # mas cerca que umbral_colision. El cono angosto (front_narrow) atrapa
        # paredes de frente que el ancho promedia y no detecta a tiempo.
        frente_bloqueado = (
            (z.front_valid and z.front < self._umbral_colision) or
            (z.front_narrow_valid and z.front_narrow < self._umbral_colision))

        if self._retrocediendo_obstaculo:
            dx = self._odom_x - self._retroceso_obstaculo_xy0[0]
            dy = self._odom_y - self._retroceso_obstaculo_xy0[1]
            if math.hypot(dx, dy) >= self._retroceso_obstaculo:
                self._publish_twist(Twist())
                self._retrocediendo_obstaculo = False
                self._esperando_obstaculo = True
                self._espera_obstaculo_inicio = self.get_clock().now()
                self._publish_event(
                    EV.COLISION,
                    f'retroceso de {self._retroceso_obstaculo * 100:.0f}cm completado'
                )
                return True
            cmd = Twist()
            cmd.linear.x = -self._v_retroceso_obstaculo
            # Retrocede en arco girando hacia la DERECHA (signo negativo,
            # mismo convenio que 'DERECHA' en _handle_girar/
            # _ejecutar_giro_ruta), no en linea recta.
            cmd.angular.z = -self._w_retroceso_obstaculo
            self._publish_twist(cmd)
            return True

        if self._esperando_obstaculo:
            if frente_bloqueado:
                self._publish_twist(Twist())
                elapsed = (
                    self.get_clock().now() - self._espera_obstaculo_inicio
                ).nanoseconds / 1e9
                if elapsed >= self._tiempo_espera_obstaculo:
                    # Se cumplio la espera y sigue bloqueado: volver a
                    # preguntar en el proximo ciclo tras otra espera igual.
                    self._espera_obstaculo_inicio = self.get_clock().now()
                return True
            self._esperando_obstaculo = False
            return False

        if frente_bloqueado:
            d_frente = z.front if z.front_valid else z.front_narrow
            self._contador_colisiones += 1
            self._publish_event(
                EV.COLISION,
                f'obstaculo a {d_frente:.2f} m cerca de {self._grid.cell}; '
                f'retrocediendo {self._retroceso_obstaculo * 100:.0f}cm'
            )
            self._retrocediendo_obstaculo = True
            self._retroceso_obstaculo_xy0 = (self._odom_x, self._odom_y)
            return True

    def _monitor_rotacion(self):
        """Anti-vuelta-completa: acumula el giro y, si supera 360 grados sin
        avanzar recto (el robot se puso a dar vueltas), corrige con un giro de
        correccion_giro_grados al lado CONTRARIO y retoma el avance."""
        if not self._correccion_giro_360 or self._terminado:
            return
        if self._state in ('ESPERA_RUTA', 'SEGUIR_RUTA', 'META', 'DETENIDO'):
            self._yaw_prev_360 = self._yaw
            self._rot_acum = 0.0
            return
        if self._yaw_prev_360 is None:
            self._yaw_prev_360 = self._yaw
            return
        # Avance recto del mapeo: resetea el acumulador (no esta girando).
        if self._state == 'AVANZAR_PARALELO':
            self._yaw_prev_360 = self._yaw
            self._rot_acum = 0.0
            return
        self._rot_acum += angle_diff(self._yaw, self._yaw_prev_360)
        self._yaw_prev_360 = self._yaw
        if self._state != 'CORREGIR_GIRO' and abs(self._rot_acum) > 2.0 * math.pi:
            self._correccion_signo = -1.0 if self._rot_acum > 0.0 else 1.0
            self._yaw_correccion0 = self._yaw
            self._rot_acum = 0.0
            self._publish_event(
                EV.GIRO,
                f'giro >360 deg detectado -> corrige '
                f'{math.degrees(self._correccion_giro_rad):.0f} deg al contrario'
            )
            self._set_state('CORREGIR_GIRO')

    def _handle_corregir_giro(self):
        """Giro corto (correccion_giro_grados) al lado contrario para
        reorientar tras una vuelta de mas de 360 grados, luego retoma el
        avance del mapeo."""
        girado = abs(angle_diff(self._yaw, self._yaw_correccion0))
        if girado >= self._correccion_giro_rad:
            self._publish_twist(Twist())
            self._begin_avanzar_paralelo()
            self._set_state('AVANZAR_PARALELO')
            return
        cmd = Twist()
        cmd.linear.x = self._v_giro_lineal
        cmd.angular.z = self._correccion_signo * self._v_giro_angular
        self._publish_twist(cmd)

        return False

    # ------------------------------------------------------------------
    # Espejo derecha/izquierda de logica_dos_reglas (seguir_pared_izquierda)
    # ------------------------------------------------------------------
    # El ajuste de linea (angulo, distancia) de una pared PARALELA al
    # pasillo tiene la MISMA relacion con el heading del robot sin
    # importar de que lado se mide -- dos paredes paralelas se ven con
    # la misma pendiente aparente desde un mismo error de heading, asi
    # que el termino de ANGULO del Kp no cambia de signo entre lados.
    # El termino de DISTANCIA si cambia: "muy cerca" siempre corrige
    # alejandose de la pared que se sigue, y alejarse es IZQUIERDA
    # cuando se sigue la derecha pero DERECHA cuando se sigue la
    # izquierda -- de ahi el signo opuesto. (Derivado geometricamente,
    # no adivinado -- ver commit que agrego este espejo.)
    def _line_valid(self, z) -> bool:
        return bool(z.left_line_valid if self._seguir_izquierda else z.right_line_valid)

    def _line_angle(self, z) -> float:
        return z.left_line_angle_rad if self._seguir_izquierda else z.right_line_angle_rad

    def _line_distance(self, z) -> float:
        return z.left_line_distance_m if self._seguir_izquierda else z.right_line_distance_m

    def _lado_valid(self, z) -> bool:
        return bool(z.left_valid if self._seguir_izquierda else z.right_valid)

    def _lado_distancia(self, z) -> float:
        return z.left if self._seguir_izquierda else z.right

    def _direccion_obstaculo(self) -> str:
        """Obstaculo al frente: gira ALEJANDOSE de la pared que se sigue."""
        return 'DERECHA' if self._seguir_izquierda else 'IZQUIERDA'

    def _direccion_vacio(self) -> str:
        """Lado seguido vacio: gira ENTRANDO al hueco de ese mismo lado."""
        return 'IZQUIERDA' if self._seguir_izquierda else 'DERECHA'

    # ------------------------------------------------------------------
    # Estados
    # ------------------------------------------------------------------
    def _handle_iniciar(self):
        self._publish_event(
            EV.INICIO, f'inicio en {self._grid.cell}, heading {self._grid.heading}'
        )
        self._tiempo_inicio = self.get_clock().now()
        self._begin_avanzar_paralelo()
        self._set_state('AVANZAR_PARALELO')

    def _begin_avanzar_paralelo(self):
        self._cell_start_xy = (self._odom_x, self._odom_y)
        self._avance_chequeo_start_xy = (self._odom_x, self._odom_y)

    def _handle_avanzar_paralelo(self):
        if self._logica_dos_reglas:
            self._handle_avanzar_paralelo_dos_reglas()
            return

        dx = self._odom_x - self._cell_start_xy[0]
        dy = self._odom_y - self._cell_start_xy[1]
        avance = math.hypot(dx, dy)

        z = self._zones
        frente_cerca = z.front_valid and z.front < self._umbral_frente_pared

        if avance >= (self._distancia_celda - self._margen_avance) or frente_cerca:
            self._publish_twist(Twist())
            self._num_celdas += 1
            self._grid.advance_cell()
            self._publish_event(
                EV.CELDA_AVANZADA, f'celda {self._grid.cell} (#{self._num_celdas})'
            )

            if self._num_celdas > self._max_celdas:
                self._publish_event(
                    EV.TIMEOUT, 'limite de celdas recorridas alcanzado sin llegar a la meta'
                )
                self._terminado = True
                self._set_state('DETENIDO')
                return

            if self._modo_simplificado:
                # Decidir con una sola lectura, sin confirmar con varias
                # muestras ni pasar por BUSCAR_PARE.
                self._derecha_libre = bool(z.right_valid and z.right > self._umbral_lado_libre)
                self._frente_libre = bool(z.front_valid and z.front > self._umbral_frente_libre)
                self._izquierda_libre = bool(z.left_valid and z.left > self._umbral_lado_libre)
                self._set_state('DECIDIR')
            else:
                self._set_state('DETECTAR_CRUCE')
            return

        self._publish_twist(self._wall_follow_cmd)

    def _handle_avanzar_paralelo_dos_reglas(self):
        """CUATRO REGLAS (ver logica_dos_reglas arriba), con AJUSTE DE
        LINEA para el lado SEGUIDO (seguir_pared_izquierda decide si es
        izquierda o derecha -- ver _line_*/_lado_*/_direccion_* arriba)
        y confirmacion de varios ciclos para el frente:

        1. Avanzar recto mientras el frente este libre.
        2. Si hay ajuste de linea valido del lado seguido, corregir
           con Kp (angulo + distancia hacia distancia_objetivo_m) --
           distingue una pared vista en diagonal (se corrige el
           angulo) de un obstaculo nuevo (no encaja como continuacion
           de esa recta).
        3. Chequeo PERIODICO del lado seguido (no deteccion continua
           por LiDAR -- en la practica no distinguia bien "pared" de
           "hueco", ver commit anterior), evaluado ANTES que el frente
           (regla 4): cada distancia_chequeo_pared_m de avance en
           linea recta (medido por odometria desde
           _avance_chequeo_start_xy), se detiene por completo y pasa a
           PAUSA_CHEQUEO_PARED, que verifica con distancia PUNTUAL si
           el lado seguido esta ocupado o vacio -- si esta vacio, gira
           90 grados DINAMICO ENTRANDO al hueco (_direccion_vacio) mas
           avance_giro_vacio_m recto, y RECIEN AHI verifica de nuevo:
           si sigue vacio repite el par giro+avance (hasta
           giro_vacio_max_repeticiones), si ya encuentra pared retoma
           AVANZAR_PARALELO (ver AVANCE_GIRO_VACIO); si esta ocupado
           desde el principio, retoma el avance reiniciando el
           contador de distancia desde ahi (evita reintentar en el
           mismo lugar).
        4. Si hay obstaculo al frente (front_narrow, cono angosto)
           sostenido durante frente_confirmaciones_ciclos seguidos, se
           detiene EN SECO y pasa por el mismo PAUSA_CHEQUEO_PARED de
           la regla 3 (detenido tiempo_chequeo_pared_s, 1s, y RECIEN
           despues verifica con distancia PUNTUAL si el lado seguido
           esta ocupado o vacio) antes de girar -- si esta VACIO, gira
           90 grados DINAMICO ENTRANDO al hueco; si esta OCUPADO, gira
           ALEJANDOSE de la pared seguida (_direccion_obstaculo, con
           self._chequeo_por_frente=True, PAUSA_CHEQUEO_PARED sabe que
           aqui no puede "retomar avance" como en la regla 3, porque
           el frente sigue bloqueado). Sin este chequeo, un giro ciego
           en un rincon angosto puede volver a encerrar al robot en el
           mismo bolsillo del que viene (loop cerrado observado en
           sim_local/, ver commit anterior). Se evalua DESPUES de la
           regla 3: en el ciclo exacto en que ambas coincidirian, se
           prioriza el chequeo periodico (el resultado es el mismo
           freno en seco).

        No cuenta celdas ni pasa por ALINEAR (el giro dinamico ya lo
        reemplaza) -- portado tal cual de
        sim_local/run_sim_laberinto.py::_correr_logica_simple.

        El color verde es solo un marcador de visualización. No forma parte
        de estas reglas y nunca modifica el estado ni el comando de velocidad.
        """
        z = self._zones

        dx = self._odom_x - self._avance_chequeo_start_xy[0]
        dy = self._odom_y - self._avance_chequeo_start_xy[1]
        avance_chequeo = math.hypot(dx, dy)

        if avance_chequeo >= self._distancia_chequeo_pared:
            self._publish_event(
                EV.GIRO, f'avanzo {avance_chequeo:.2f}m -> detenido a verificar pared'
            )
            self._chequeo_por_frente = False
            self._publish_twist(Twist())
            self._pausa_chequeo_start = self.get_clock().now()
            self._set_state('PAUSA_CHEQUEO_PARED')
            return

        frente_cerca_1_ciclo = z.front_narrow_valid and z.front_narrow < self._umbral_frente_pared
        self._contador_frente_dos_reglas = (
            self._contador_frente_dos_reglas + 1 if frente_cerca_1_ciclo else 0
        )

        if self._contador_frente_dos_reglas >= self._frente_confirmaciones_ciclos:
            self._contador_frente_dos_reglas = 0
            self._publish_event(
                EV.GIRO, f'obstaculo al frente ({z.front_narrow:.2f}m) -> detenido a verificar pared'
            )
            self._chequeo_por_frente = True
            self._publish_twist(Twist())
            self._pausa_chequeo_start = self.get_clock().now()
            self._set_state('PAUSA_CHEQUEO_PARED')
            return

        cmd = Twist()
        if not self._line_valid(z):
            # Perdida no confirmada todavia (podria ser un solo
            # vistazo de ruido): avanzar recto, sin corregir nada.
            cmd.linear.x = self._velocidad_recta
            self._publish_twist(cmd)
            return
        # Termino de ANGULO: mismo signo sin importar el lado (dos
        # paredes paralelas se ven con la misma pendiente aparente
        # desde un mismo error de heading). Termino de DISTANCIA:
        # signo opuesto segun el lado (alejarse de la pared seguida es
        # IZQUIERDA si se sigue la derecha, DERECHA si se sigue la
        # izquierda) -- ver nota larga junto a _line_*/_lado_* arriba.
        signo_distancia = -1.0 if self._seguir_izquierda else 1.0
        error_distancia = self._distancia_objetivo_recta - self._line_distance(z)
        correccion = (self._ganancia_angulo_recta * self._line_angle(z)
                      + signo_distancia * self._ganancia_distancia_recta * error_distancia)
        # Anti-choque lateral (LiDAR): si la pared seguida esta MUY cerca
        # (< umbral_lateral_min), fuerza el giro maximo alejandose para no
        # rozarla, sin confiar solo en el termino Kp.
        if self._lado_valid(z) and self._lado_distancia(z) < self._umbral_lateral_min:
            correccion = signo_distancia * self._angular_max_recta
        cmd.linear.x = self._velocidad_recta
        cmd.angular.z = max(-self._angular_max_recta, min(self._angular_max_recta, correccion))
        self._publish_twist(cmd)

    def _handle_pausa_chequeo_pared(self):
        """Detenido tiempo_chequeo_pared_s (0.5s) -- ya sea por el
        chequeo PERIODICO (regla 3) o porque se confirmo un obstaculo
        al frente (regla 4, self._chequeo_por_frente=True) -- y RECIEN
        despues verifica con distancia PUNTUAL (no el ajuste de linea)
        si el lado SEGUIDO esta ocupado (pared) o vacio:

        - Si esta VACIO: gira ENTRANDO al hueco (_direccion_vacio, en
          ambos casos).
        - Si esta OCUPADO:
          - Si vino del chequeo periodico (regla 3): retoma el avance
            normal, reiniciando el contador de distancia desde aqui
            (evita volver a dispararse de inmediato en el mismo
            lugar).
          - Si vino de un obstaculo al frente (regla 4): NO puede
            simplemente retomar el avance (el frente sigue bloqueado)
            -- gira ALEJANDOSE de la pared seguida (_direccion_
            obstaculo).

        "Vacio" se confirma con chequeo_pared_confirmaciones_ciclos
        lecturas SEGUIDAS (no una sola) -- un giro es una decision
        cara de revertir, asi que se exige sostener el "vacio" varios
        ciclos (sigue detenido mientras confirma) antes de
        comprometerse. "Ocupado" no necesita esta confirmacion (el
        peor caso es solo seguir derecho un poco mas)."""
        self._publish_twist(Twist())
        elapsed = (self.get_clock().now() - self._pausa_chequeo_start).nanoseconds / 1e9
        if elapsed < self._tiempo_chequeo_pared:
            return

        z = self._zones
        lado_libre = bool(self._lado_valid(z) and self._lado_distancia(z) > self._umbral_lado_libre)

        if lado_libre:
            self._contador_derecha_libre += 1
            if self._contador_derecha_libre < self._chequeo_pared_confirmaciones_ciclos:
                return
            self._contador_derecha_libre = 0
            self._decision_actual = self._direccion_vacio()
            self._yaw_inicio_giro = self._yaw
            self._giro_vacio_fase = 1
            self._giro_vacio_repeticiones = 0
            self._contador_giros_fisicos += 1
            self._publish_event(
                EV.GIRO, f'lado seguido vacio ({self._lado_distancia(z):.2f}m) -> {self._decision_actual}'
            )
            self._set_state('GIRAR')
            return

        self._contador_derecha_libre = 0

        if self._chequeo_por_frente:
            self._decision_actual = self._direccion_obstaculo()
            self._yaw_inicio_giro = self._yaw
            self._giro_vacio_fase = 0
            self._contador_giros_fisicos += 1
            self._publish_event(
                EV.GIRO, f'lado seguido ocupado, frente bloqueado -> {self._decision_actual}'
            )
            self._set_state('GIRAR')
            return

        # Ocupado (chequeo periodico): sigue habiendo pared -- retoma
        # el avance normal, reiniciando el contador de distancia.
        self._publish_event(EV.GIRO, 'lado seguido ocupado -> retoma avance')
        self._avance_chequeo_start_xy = (self._odom_x, self._odom_y)
        self._set_state('AVANZAR_PARALELO')

    def _handle_detectar_cruce(self):
        self._publish_twist(Twist())

        if self._cruce_muestras is None:
            self._cruce_muestras = {'right': [], 'front': [], 'left': []}

        z = self._zones
        self._cruce_muestras['right'].append(
            bool(z.right_valid and z.right > self._umbral_lado_libre)
        )
        self._cruce_muestras['front'].append(
            bool(z.front_valid and z.front > self._umbral_frente_libre)
        )
        self._cruce_muestras['left'].append(
            bool(z.left_valid and z.left > self._umbral_lado_libre)
        )

        if len(self._cruce_muestras['right']) < self._muestras_confirmacion:
            return

        def consenso(muestras):
            return sum(muestras) >= self._consenso_minimo

        self._derecha_libre = consenso(self._cruce_muestras['right'])
        self._frente_libre = consenso(self._cruce_muestras['front'])
        self._izquierda_libre = consenso(self._cruce_muestras['left'])
        self._cruce_muestras = None

        self._publish_event(
            EV.CRUCE,
            f'derecha={self._derecha_libre} frente={self._frente_libre} '
            f'izquierda={self._izquierda_libre}',
        )

        self._buscar_pare_start = self.get_clock().now()
        self._pare_hold_start = None
        self._set_state('BUSCAR_PARE')

    def _handle_buscar_pare(self):
        self._publish_twist(Twist())

        if not self._usar_camara:
            self._set_state('DECIDIR')
            return

        cell = self._grid.cell

        # Si ya se inicio el conteo de los 3 s, completarlo sin importar
        # parpadeos momentaneos de la deteccion (evita abortar el PARE
        # a mitad de camino si la camara pierde el color rojo un frame).
        if self._pare_hold_start is not None:
            elapsed = (self.get_clock().now() - self._pare_hold_start).nanoseconds / 1e9
            if elapsed >= self._tiempo_pare:
                self._celdas_pare_respetadas.add(cell)
                self._publish_event(EV.PARE_RESPETADO, f'PARE respetado en {cell}')
                self._set_state('DECIDIR')
            return

        if self._pare_activo and cell not in self._celdas_pare_respetadas:
            self._publish_event(EV.PARE_DETECTADO, f'senal PARE detectada en {cell}')
            self._pare_hold_start = self.get_clock().now()
            return

        elapsed_settle = (self.get_clock().now() - self._buscar_pare_start).nanoseconds / 1e9
        if elapsed_settle >= self._tiempo_espera_camara:
            self._set_state('DECIDIR')

    def _handle_decidir(self):
        if self._derecha_libre:
            direction = 'DERECHA'
        elif self._frente_libre:
            direction = 'NINGUNO'
        elif self._izquierda_libre:
            direction = 'IZQUIERDA'
        else:
            direction = 'ATRAS'
            self._publish_event(EV.DEAD_END, f'callejon sin salida en {self._grid.cell}')

        self._decision_actual = direction

        if direction == 'NINGUNO':
            if self._modo_simplificado:
                self._begin_avanzar_paralelo()
                self._set_state('AVANZAR_PARALELO')
            else:
                self._alinear_start = None
                self._set_state('ALINEAR')
            return

        self._giro_objetivo = self._compute_turn_target(self._yaw, direction)
        self._publish_event(EV.GIRO, f'{direction} desde {self._grid.cell}')
        self._publish_twist(Twist())
        self._pausa_giro_start = self.get_clock().now()
        self._set_state('PAUSA_GIRO')

    def _handle_pausa_giro(self):
        """Robot detenido ``tiempo_pausa_antes_girar_s`` antes de arrancar
        el arco de GIRAR -- separa visiblemente "termine de avanzar" de
        "empiezo a girar" en vez de una transicion instantanea."""
        self._publish_twist(Twist())
        elapsed = (self.get_clock().now() - self._pausa_giro_start).nanoseconds / 1e9
        if elapsed >= self._tiempo_pausa_antes_girar:
            self._set_state('GIRAR')

    def _compute_turn_target(self, yaw: float, direction: str) -> float:
        if direction == 'DERECHA':
            delta = -self._angulo_giro_rad
        elif direction == 'IZQUIERDA':
            delta = self._angulo_giro_rad
        elif direction == 'ATRAS':
            delta = math.pi
        else:
            delta = 0.0
        return normalize_angle(yaw + delta)

    def _handle_girar(self):
        if self._logica_dos_reglas:
            self._handle_girar_dinamico()
            return

        error = angle_diff(self._giro_objetivo, self._yaw)

        if abs(error) <= self._tolerancia_giro_rad:
            self._publish_twist(Twist())
            self._grid.apply_turn(self._decision_actual)
            # ALINEAR corre siempre, incluso en modo_simplificado: GIRAR
            # por si solo solo cierra el lazo contra el yaw de odometria
            # (un angulo objetivo fijo, con la deriva propia del
            # odometro pese al factor de correccion). ALINEAR corrige
            # ese resultado con el LiDAR real (right_front/right_rear)
            # despues del giro -- es el feedback real, no un angulo fijo.
            self._alinear_start = None
            self._set_state('ALINEAR')
            return

        # Chasis Ackermann: no puede rotar en el sitio. Se aproxima el
        # giro con avance lento + direccion maxima, cerrando el lazo
        # con el yaw de la odometria (no con tiempo fijo).
        cmd = Twist()
        cmd.linear.x = self._v_giro_lineal
        cmd.angular.z = self._v_giro_angular if error > 0.0 else -self._v_giro_angular
        self._publish_twist(cmd)

    def _handle_girar_dinamico(self):
        """Giro FIJO por odometria (logica_dos_reglas). Antes giraba
        DINAMICO: paraba en cuanto el LiDAR mostraba pared paralela
        (right/left_line_angle_rad ~0). En pista real eso resultaba en
        giros inconsistentes -- a veces bastante menos de 90 grados, a
        veces bastante mas -- porque "parece paralelo" segun el LiDAR
        puede dispararse antes o despues de los 90 grados reales,
        segun ruido o la geometria puntual del rincon.

        Ahora cierra el lazo SOLO contra el yaw de odometria
        (self._yaw, ya con factor_ang_odom aplicado -- calibrado en
        pista) hasta angulo_giro_deg (90 grados), igual que el giro
        fijo original -- exacto y repetible, sin depender de que el
        LiDAR encuentre algo parecido a "paralelo" en el momento
        justo. angulo_maximo_giro_deg se mantiene como tope de
        seguridad adicional (por si el odometro se traba y nunca
        llega al objetivo).

        Si self._giro_vacio_fase == 1 (este giro es parte de una
        secuencia de "lado seguido vacio"), no vuelve directo a
        AVANZAR_PARALELO al terminar -- pasa por AVANCE_GIRO_VACIO
        (avance_giro_vacio_m recto), que verifica de nuevo el lado
        seguido: si sigue vacio, repite otro giro de 90 (hasta
        giro_vacio_max_repeticiones, tope de seguridad); si ya
        encuentra pared, recien ahi retoma AVANZAR_PARALELO."""
        angulo_girado = abs(angle_diff(self._yaw, self._yaw_inicio_giro))

        if angulo_girado >= self._angulo_giro_rad or angulo_girado >= self._angulo_maximo_giro_rad:
            self._publish_twist(Twist())
            self._grid.apply_turn(self._decision_actual)
            # Grabacion pasiva del giro para la ruta corta (no altera el mapeo).
            if self._ruta_activa and self._explorer is not None:
                self._explorer.girar(self._decision_actual)
                self._ruta_dist_acc = 0.0
                self._ruta_last_xy = (self._odom_x, self._odom_y)
            self.get_logger().info(
                f'GIRO TERMINADO (90 fijo): girado={math.degrees(angulo_girado):.0f} deg'
            )
            # Deteccion de patinaje: compara lo girado por odometria de
            # rueda contra lo girado por el giroscopio (IMU, no depende de
            # las ruedas) durante este mismo giro. Una diferencia grande
            # indica que una rueda patino (giro real distinto del medido
            # por las ruedas).
            odom_deg = math.degrees(angulo_girado)
            imu_deg = abs(math.degrees(self._imu_acum_giro))
            diff_deg = abs(odom_deg - imu_deg)
            if diff_deg > self._umbral_patinaje:
                self._publish_event(
                    EV.PATINAJE,
                    f'posible patinaje: odom={odom_deg:.0f}° imu={imu_deg:.0f}° '
                    f'diff={diff_deg:.0f}°'
                )
            if self._giro_vacio_fase == 1:
                self._avance_fijo_inicio_xy = (self._odom_x, self._odom_y)
                self._set_state('AVANCE_GIRO_VACIO')
                return
            self._begin_avanzar_paralelo()
            self._set_state('AVANZAR_PARALELO')
            return

        cmd = Twist()
        cmd.linear.x = self._v_giro_lineal
        cmd.angular.z = self._v_giro_angular if self._decision_actual == 'IZQUIERDA' else -self._v_giro_angular
        self._publish_twist(cmd)

    def _handle_avance_giro_vacio(self):
        """Tras un giro de 90 de la secuencia de "lado seguido vacio",
        avanza avance_giro_vacio_m (12cm) en linea recta y RECIEN
        despues verifica de nuevo con distancia PUNTUAL si el lado
        seguido sigue vacio -- si sigue vacio, repite: otro giro de 90
        en la MISMA direccion (self._decision_actual no cambia) mas
        otro avance de 12cm, hasta giro_vacio_max_repeticiones (tope
        de seguridad, evita girar para siempre en un espacio muy
        abierto). Si ya encuentra pared, retoma AVANZAR_PARALELO."""
        dx = self._odom_x - self._avance_fijo_inicio_xy[0]
        dy = self._odom_y - self._avance_fijo_inicio_xy[1]
        avance = math.hypot(dx, dy)

        if avance < self._avance_giro_vacio:
            cmd = Twist()
            cmd.linear.x = self._velocidad_recta
            self._publish_twist(cmd)
            return

        self._publish_twist(Twist())
        z = self._zones
        lado_libre = bool(self._lado_valid(z) and self._lado_distancia(z) > self._umbral_lado_libre)

        if lado_libre and self._giro_vacio_repeticiones < self._giro_vacio_max_repeticiones:
            self._giro_vacio_repeticiones += 1
            self._yaw_inicio_giro = self._yaw
            self._contador_giros_fisicos += 1
            self._publish_event(
                EV.GIRO,
                f'avanzo {avance:.2f}m, lado seguido sigue vacio '
                f'-> otro giro (rep {self._giro_vacio_repeticiones})'
            )
            self._set_state('GIRAR')
            return

        self._giro_vacio_fase = 0
        motivo = 'lado seguido ocupado' if not lado_libre else 'tope de repeticiones'
        self._publish_event(EV.GIRO, f'avanzo {avance:.2f}m, {motivo} -> retoma avance')
        self._begin_avanzar_paralelo()
        self._set_state('AVANZAR_PARALELO')

    def _handle_alinear(self):
        if self._alinear_start is None:
            self._alinear_start = self.get_clock().now()

        z = self._zones
        if not (z.right_front_valid and z.right_rear_valid):
            # Sin pared derecha de referencia (p.ej. abertura tras el
            # giro): el yaw de GIRAR ya dejo al robot orientado al
            # cardinal correcto, se continua sin correccion adicional.
            self._alinear_start = None
            self._set_state('VERIFICAR_META')
            return

        error_angulo = z.right_front - z.right_rear
        elapsed = (self.get_clock().now() - self._alinear_start).nanoseconds / 1e9

        if abs(error_angulo) <= self._tolerancia_alineacion or elapsed >= self._tiempo_max_alinear:
            self._publish_twist(Twist())
            self._alinear_start = None
            self._set_state('VERIFICAR_META')
            return

        cmd = Twist()
        cmd.linear.x = self._v_alinear_lineal
        cmd.angular.z = -self._v_alinear_angular if error_angulo > 0.0 else self._v_alinear_angular
        self._publish_twist(cmd)

    def _handle_verificar_meta(self):
        if self._grid.cell == self._celda_meta:
            self._publish_twist(Twist())
            self._publish_event(EV.META, f'meta alcanzada en {self._grid.cell}')
            self._terminado = True
            self._set_state('META')
            return

        self._begin_avanzar_paralelo()
        self._set_state('AVANZAR_PARALELO')

    # ------------------------------------------------------------------
    # FASE 2: ruta corta (BFS) + freno + manejo por guion de movimientos
    # ------------------------------------------------------------------
    def _registrar_avance_ruta(self):
        """Discretiza el avance recto del MAPEO en celdas y graba el grafo.

        Solo acumula distancia mientras el mapeo avanza recto
        (AVANZAR_PARALELO); en cualquier otro estado re-ancla la referencia
        para no contar los arcos de giro. Cada tamano_celda_m recorridos
        registra un avance de celda y revisa si el robot regreso a la base.
        No modifica ninguna variable del mapeo.
        """
        if not self._ruta_activa or self._explorer is None or self._terminado:
            return
        if self._state in ('ESPERA_RUTA', 'SEGUIR_RUTA'):
            return   # detenido esperando, o manejando: no grabar
        # Cuenta avances de celda (solo en el avance recto del mapeo). El carrito
        # NO frena solo al volver al inicio: solo frena con /maze/calcular_ruta.
        if self._state != 'AVANZAR_PARALELO':
            self._ruta_last_xy = (self._odom_x, self._odom_y)
            return
        if self._ruta_last_xy is None:
            self._ruta_last_xy = (self._odom_x, self._odom_y)
            return
        dx = self._odom_x - self._ruta_last_xy[0]
        dy = self._odom_y - self._ruta_last_xy[1]
        self._ruta_last_xy = (self._odom_x, self._odom_y)
        self._ruta_dist_acc += math.hypot(dx, dy)
        while self._ruta_dist_acc >= self._tamano_celda:
            self._ruta_dist_acc -= self._tamano_celda
            self._explorer.avanzar()

    def _calcular_ruta(self) -> bool:
        """Calcula la ruta corta (BFS) y la publica (amarillo). NO mueve el
        carrito ni cambia de estado. Devuelve True si obtuvo una ruta valida."""
        if self._meta_cell is None:
            self._publish_event(
                EV.TIMEOUT, 'sin ruta: aun no se vio la meta (verde)')
            return False
        ruta = bfs_ruta(self._ruta_start_cell, self._meta_cell,
                        self._explorer.aristas)
        if not ruta or len(ruta) < 2:
            self._publish_event(
                EV.TIMEOUT, f'sin ruta valida a la meta {self._meta_cell}')
            return False
        self._ruta_celdas = ruta
        # Rumbo de arranque del guion. Por defecto (ruta_asume_rumbo_inicial=
        # True) se asume que el carrito estara en el INICIO mirando
        # ruta_heading_inicial (NORTE) -- el punto de partida es siempre el
        # mismo. Si es False, se usa el rumbo interno de odometria.
        rumbo_arranque = (self._ruta_heading_inicial
                          if self._ruta_asume_rumbo_inicial
                          else self._explorer.heading)
        self._ruta_movimientos = celdas_a_movimientos(ruta, rumbo_arranque)
        self._ruta_idx = 0
        self._ruta_fase = 'GIRO'
        self._ruta_giro_restante = None
        self._publicar_ruta()
        self._publish_event(
            EV.META, f'ruta corta calculada: {len(ruta)} celdas')
        return True

    def _publicar_ruta(self):
        self._ruta_json = json.dumps({
            'listo': True,
            'celdas': [[int(c), int(r)] for (c, r) in self._ruta_celdas],
            'meta': [int(self._meta_cell[0]), int(self._meta_cell[1])],
        })
        self._ruta_pub.publish(String(data=self._ruta_json))

    def _republicar_ruta(self):
        """Re-emite la ruta cada ~1 s para visualizadores que se conecten
        tarde (el publisher es volatil)."""
        if self._ruta_json is None:
            return
        self._ruta_pub_counter += 1
        if self._ruta_pub_counter >= int(max(1.0, self._control_rate_hz)):
            self._ruta_pub_counter = 0
            self._ruta_pub.publish(String(data=self._ruta_json))

    def _handle_espera_ruta(self):
        """Detenido con la ruta ya calculada (amarillo pintado). Espera el
        comando /maze/iniciar_ruta para PARTIR. No se mueve ni retoma el mapeo
        por su cuenta: el carrito nunca arranca solo."""
        self._publish_twist(Twist())
        self._republicar_ruta()

    def _handle_seguir_ruta(self):
        """Maneja la ruta corta obedeciendo SOLO el guion de movimientos.

        No sigue pared ni ajusta linea: gira lo que dice la ruta (giro fijo
        de 90/180 cerrado por yaw) y avanza en recto las celdas indicadas
        (cerrado por odometria). La anticolision es la unica capa reactiva y
        ya corre global en _on_timer antes de este handler.
        """
        self._republicar_ruta()
        if (self._ruta_movimientos is None
                or self._ruta_idx >= len(self._ruta_movimientos)):
            self._publish_twist(Twist())
            self._publish_event(
                EV.META,
                'ruta corta completada; DETENIDO. Reenvia /maze/iniciar_ruta '
                'para repetir (colocalo en el inicio mirando NORTE).'
            )
            self._set_state('ESPERA_RUTA')   # vuelve a esperar -> se puede repetir
            return

        mov = self._ruta_movimientos[self._ruta_idx]
        if self._ruta_fase == 'GIRO':
            self._ejecutar_giro_ruta(mov['giro'])
        else:
            self._ejecutar_avance_ruta(mov['celdas'])

    def _iniciar_avance_ruta(self):
        self._ruta_fase = 'AVANCE'
        self._ruta_avance_xy0 = (self._odom_x, self._odom_y)

    def _ejecutar_giro_ruta(self, direccion: str):
        """Giro fijo por yaw. Un giro de 180 (ATRAS) se hace como DOS sub-giros
        de 90 hacia la izquierda: cerrar directo a 180 con abs(angle_diff) es
        fragil (solo se cumple exacto en pi; el muestreo discreto se pasa y
        angle_diff decrece -> giro infinito). Cada sub-giro reusa el cierre a
        90 (angulo_giro_rad), que si es monotono y confiable."""
        if direccion == 'NINGUNO':
            self._iniciar_avance_ruta()
            return
        if self._ruta_giro_restante is None:
            self._ruta_giro_restante = 2 if direccion == 'ATRAS' else 1
            self._ruta_giro_yaw0 = self._yaw
        girado = abs(angle_diff(self._yaw, self._ruta_giro_yaw0))
        if girado >= self._angulo_giro_rad:
            self._publish_twist(Twist())
            self._ruta_giro_restante -= 1
            if self._ruta_giro_restante > 0:
                self._ruta_giro_yaw0 = self._yaw   # arrancar el siguiente 90
                return
            self._ruta_giro_restante = None
            self._iniciar_avance_ruta()
            return
        cmd = Twist()
        cmd.linear.x = self._v_giro_lineal
        izquierda = direccion in ('IZQUIERDA', 'ATRAS')
        cmd.angular.z = self._v_giro_angular if izquierda else -self._v_giro_angular
        self._publish_twist(cmd)

    def _ejecutar_avance_ruta(self, celdas: int):
        dx = self._odom_x - self._ruta_avance_xy0[0]
        dy = self._odom_y - self._ruta_avance_xy0[1]
        objetivo = celdas * self._tamano_celda
        if math.hypot(dx, dy) >= objetivo:
            self._publish_twist(Twist())
            self._ruta_idx += 1
            self._ruta_fase = 'GIRO'
            self._ruta_giro_restante = None
            return
        cmd = Twist()
        cmd.linear.x = self._velocidad_recta
        self._publish_twist(cmd)

    def _handle_meta(self):
        self._publish_twist(Twist())

    def _handle_detenido(self):
        self._publish_twist(Twist())

    # ------------------------------------------------------------------
    # Utilidades de publicacion
    # ------------------------------------------------------------------
    def _metricas_actuales(self) -> dict:
        """Campos Gran Prix que espera metrics_logger.py (dead_ends_visitados
        no se calcula aca -- sin una nocion clara de "callejon sin salida"
        en logica_dos_reglas, se deja el default 0 de metrics_logger antes
        que inventar un numero)."""
        if self._tiempo_inicio is not None:
            tiempo_s = (self.get_clock().now() - self._tiempo_inicio).nanoseconds / 1e9
        else:
            tiempo_s = 0.0
        return {
            'llego_meta': self._meta_cell is not None,
            'tiempo_s': round(tiempo_s, 1),
            'long_ruta_cm': round(self._distancia_total_m * 100.0, 1),
            'colisiones': self._contador_colisiones,
            'pare_detectados': self._contador_pare_detectados,
            'pare_respetados': self._contador_pare_respetados,
        }

    def _publish_twist(self, cmd: Twist):
        self._cmd_pub.publish(cmd)
        z = self._zones
        if z is not None:
            followed = z.left_line_distance_m if self._seguir_izquierda else z.right_line_distance_m
            followed_valid = z.left_line_valid if self._seguir_izquierda else z.right_line_valid
            rear = z.left if self._seguir_izquierda else z.right
            rear_valid = z.left_valid if self._seguir_izquierda else z.right_valid
            payload = {
                'estado': self._state,
                'v': float(cmd.linear.x),
                'w': float(cmd.angular.z),
                'd_frente': float(z.front) if z.front_valid else None,
                'd_atras': None,
                'd_izq': float(z.left) if z.left_valid else None,
                'd_der': float(z.right) if z.right_valid else None,
                'd_lado_frontal': float(followed) if followed_valid else None,
                'd_lado_trasera': float(rear) if rear_valid else None,
                # Un giro fisico de 90 real (los 3 puntos donde se decide
                # direccion y se entra a GIRAR) -- a diferencia de tramo
                # (que cuenta visualizador_web via /maze/estado), esto NO
                # se pierde cuando un "giro vacio" encadena varias vueltas
                # de 90 sin volver a pasar por AVANZAR_PARALELO entre medio
                # (ver AVANCE_GIRO_VACIO). visualizador_web/index.html usan
                # este contador para saber en que vertice de la ruta fija
                # dibujada esta el carrito, sin recalcular nada.
                'giros_fisicos': self._contador_giros_fisicos,
            }
            # Congelado en META (ver _on_verde) o en vivo mientras mapea --
            # metrics_logger.py lee estos mismos campos de este topico.
            payload.update(self._metricas_meta or self._metricas_actuales())
            self._metrics_pub.publish(String(data=json.dumps(payload)))

    def _publish_event(self, tipo: str, detalle: str):
        self._event_pub.publish(String(data=json.dumps({'tipo': tipo, 'detalle': detalle})))
        self.get_logger().info(f'[{tipo}] {detalle}')

    def _set_state(self, new_state: str):
        # Sin log de consola aqui a proposito -- cada transicion
        # relevante ya imprime una sola linea con el detalle via
        # _publish_event (o get_logger().info directo en GIRO
        # TERMINADO), asi que loguear tambien la transicion de estado
        # en si duplicaba la info. El topico /robot_state (para otros
        # nodos/herramientas) se sigue publicando igual.
        if new_state == 'GIRAR' and self._state != 'GIRAR':
            self._imu_acum_giro = 0.0
            self._imu_t_prev = self.get_clock().now()
        self._state = new_state
        self._state_pub.publish(String(data=self._state))


def main(args=None):
    rclpy.init(args=args)
    node = MazeSolverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
