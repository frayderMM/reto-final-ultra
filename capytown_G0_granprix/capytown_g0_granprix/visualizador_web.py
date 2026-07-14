#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualizador_web.py - EMISOR DE DATOS del visualizador web (corre en el CARRITO).

Arquitectura (para no cargar la Pi ni depender de VNC):
  · CARRITO (Pi): este nodo se suscribe a los tópicos ROS y EMITE los datos por
    HTTP como JSON ligero en  GET /data  (con CORS habilitado). No dibuja nada.
  · LAPTOP:       abre  web/index.html  (frontend puro con Canvas) que hace
    fetch a  http://<IP-de-la-Pi>:8080/data  y DIBUJA todo en el navegador.

Así la Pi solo serializa JSON (carga casi nula) y el render pesado ocurre en la
laptop. Reemplaza al visualizador de matplotlib.

Uso (en el carrito):
  ros2 run capytown_g0_granprix visualizador_web
En la laptop: abrir web/index.html y poner la IP de la Pi (def. 10.42.0.1).
"""

import json
import math
import base64
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu, Image
from std_msgs.msg import Bool, String

try:
    import cv2
    import numpy as np
    OPENCV_DISPONIBLE = True
except Exception:
    cv2 = None
    np = None
    OPENCV_DISPONIBLE = False

# ── Constantes de la transformación odom→pista ─────────────────────────────
CELDA_M = 0.30
INICIO_CELDA = (0, 7)            # A7 (centro del robot; la zona de INICIO es A8)
INICIO_POS = (INICIO_CELDA[0] + 0.5, INICIO_CELDA[1] - 0.5)

# El cartel META solo es válido en los bloques finales mostrados en el mapa:
# columnas J, K y L; filas 1 y 2. Las coordenadas internas son base cero.
META_COLUMNAS_VALIDAS = frozenset((9, 10, 11))
META_FILAS_VALIDAS = frozenset((0, 1))

TRAY_PASO_MIN_M = 0.02           # decimado del recorrido
TRAY_MAX_PTS = 4000


def _norm(a):
    return math.atan2(math.sin(a), math.cos(a))


def _imagen_a_jpeg_data_url(msg: Image, max_width: int, calidad: int = 70):
    """Convierte un sensor_msgs/Image a data URL JPEG para el navegador."""
    encoding = (msg.encoding or '').lower()
    if encoding in ('mjpeg', 'jpeg'):
        return 'data:image/jpeg;base64,' + base64.b64encode(msg.data).decode('ascii')

    if not OPENCV_DISPONIBLE:
        return None

    try:
        h, w = int(msg.height), int(msg.width)
        if h <= 0 or w <= 0:
            return None

        if encoding in ('rgb8', 'bgr8'):
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))
            if encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif encoding == 'mono8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif encoding in ('yuyv', 'yuv422_yuy2'):
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 2))
            img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
        else:
            return None

        max_width = max(80, int(max_width))
        if w > max_width:
            scale = max_width / float(w)
            img = cv2.resize(img, (max_width, max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), calidad])
        if not ok:
            return None
        return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')
    except Exception:
        return None


class VisualizadorWeb(Node):
    def __init__(self):
        super().__init__('visualizador_web')
        self.declare_parameter('puerto', 8080)
        self.declare_parameter('front_lidar', 180.0)   # grados (MS200: cable atrás)
        self.declare_parameter('lidar_fov_deg', 180.0) # solo frente + laterales
        self.declare_parameter('factor_dist_odom', 0.9474)
        self.declare_parameter('factor_ang_odom', 0.9899)
        self.declare_parameter('factor_ang_imu', 1.0)
        self.declare_parameter('fuente_yaw', 'odom')        # odom|imu
        self.declare_parameter('topic_imu', '/imu')
        self.declare_parameter('imu_fuente', 'gyro')        # gyro|orientacion|auto
        self.declare_parameter('track_yaw_sign', 1.0)
        self.declare_parameter('track_snap_heading', True)
        self.declare_parameter('track_lidar_init_m', 0.0)
        self.declare_parameter('track_lidar_blend', 0.0)
        self.declare_parameter('topic_camera', '/image_raw')
        self.declare_parameter('camera_max_width', 360)
        # 15 FPS evita saturar la Pi y la red con JPEG/base64; la cámara física
        # puede seguir publicando a 30 FPS para el detector de colores.
        self.declare_parameter('camera_fps', 15.0)
        self.declare_parameter('camera_jpeg_quality', 45)
        self.puerto = int(self.get_parameter('puerto').value)
        self.front_rad = math.radians(self.get_parameter('front_lidar').value)
        self.lidar_fov_rad = math.radians(self.get_parameter('lidar_fov_deg').value)
        self.factor_dist_odom = float(self.get_parameter('factor_dist_odom').value)
        self.factor_ang_odom = float(self.get_parameter('factor_ang_odom').value)
        self.factor_ang_imu = float(self.get_parameter('factor_ang_imu').value)
        self.fuente_yaw = str(self.get_parameter('fuente_yaw').value).lower()
        if self.fuente_yaw not in ('odom', 'imu'):
            self.fuente_yaw = 'odom'
        self.topic_imu = self.get_parameter('topic_imu').value
        self.imu_fuente = str(self.get_parameter('imu_fuente').value).lower()
        self.track_yaw_sign = float(self.get_parameter('track_yaw_sign').value)
        self.track_snap_heading = bool(self.get_parameter('track_snap_heading').value)
        self.track_lidar_init_m = float(self.get_parameter('track_lidar_init_m').value)
        self.track_lidar_blend = float(self.get_parameter('track_lidar_blend').value)
        self.topic_camera = self.get_parameter('topic_camera').value
        self.camera_max_width = int(self.get_parameter('camera_max_width').value)
        self.camera_fps = float(self.get_parameter('camera_fps').value)
        self.camera_jpeg_quality = int(self.get_parameter('camera_jpeg_quality').value)
        self._camera_period = 1.0 / self.camera_fps if self.camera_fps > 0.0 else 0.0

        # ── Estado compartido (protegido por lock) ───────────────────────
        self.lock = threading.Lock()
        self.estado = 'INICIO'
        self.tramo = 0
        self.giros_fisicos = 0
        self.odom_cm = 0.0
        self.lidar = []                 # [[x,y], ...] marco robot
        self._lidar_full = []           # puntos 360 para anclar inicio
        self._wall_pts = []             # puntos de paredes detectadas
        self._wall_segments = []        # paredes detectadas como polilineas
        self._calib_pts = []            # puntos usados para calibrar, pintados azul
        self.d_frente = None
        self.d_atras = None
        self.d_izq = None
        self.d_der = None
        self.d_lado_frontal = None
        self.d_lado_trasera = None
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.traj = deque(maxlen=TRAY_MAX_PTS)   # [[px,py], ...] coords pista
        self.pare = deque(maxlen=200)
        self._pare_activo = False
        self.verde_detectado = False
        self._verde_activo = False
        self._meta_registrada = False
        self.verde_pos = deque(maxlen=1)
        self.ruta_corta = []            # [[px,py], ...] ruta amarilla (fase 2)
        self.pos = None                 # [px,py] coords pista
        self.head = [0.0, 0.0]          # rumbo en coords pista
        self._odom_origen = None        # (x0,y0,yaw0)  yaw0 de fuente_yaw
        self._ultimo_odom = None
        self._ultimo_yaw = None         # último yaw de ODOM (solo signo avance/retroceso)
        self.imu_yaw = None             # yaw del IMU si fuente_yaw='imu'
        self._imu_yaw_int = 0.0         # acumulador si el IMU no fusiona orientación
        self._t_imu_prev = None
        self._ultimo_tray = None
        self._track_pos = [INICIO_POS[0], INICIO_POS[1]]
        self._calib_info = {'activa': True, 'izq_m': None, 'atras_m': None}
        self.camera_src = None
        self.camera_info = {'ok': False, 'topic': self.topic_camera,
                            'encoding': None, 'w': 0, 'h': 0}
        self._camera_last_emit_s = 0.0

        # ── ROS I/O ───────────────────────────────────────────────────────
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(LaserScan, '/scan', self.cb_scan, qos)
        self.create_subscription(Odometry, '/odom_raw', self.cb_odom, qos)
        self.create_subscription(Imu, self.topic_imu, self.cb_imu, qos)
        self.create_subscription(Image, self.topic_camera, self.cb_camera, qos)
        self.create_subscription(Bool, '/pare_detectado', self.cb_pare, 10)
        self.create_subscription(Bool, '/verde_detectado', self.cb_verde, 10)
        self.create_subscription(String, '/maze/estado', self.cb_estado, 10)
        self.create_subscription(String, '/maze/metricas', self.cb_metricas, 10)
        self.create_subscription(String, '/maze/ruta_corta', self.cb_ruta, 10)

        # ── Servidor HTTP en hilo aparte ──────────────────────────────────
        self._arrancar_http()
        self.get_logger().warn(
            f'Emisor web activo en el puerto {self.puerto}. En la laptop abre '
            f'web/index.html y usa la IP de esta Pi (GET /data).')

    # ── Transformaciones odom → pista ───────────────────────────────────
    def _yaw_rel_pista(self, yaw, factor_ang):
        """Yaw relativo al arranque, corregido para el sentido real."""
        return _norm((yaw - self._odom_origen[2]) *
                     factor_ang * self.track_yaw_sign)

    def _rumbo_desde_rel(self, yaw_rel, snap=False):
        """Vector unitario en coordenadas de pista: yaw_rel=0 apunta hacia arriba."""
        if snap:
            yaw_rel = round(yaw_rel / (math.pi / 2.0)) * (math.pi / 2.0)
        return -math.sin(yaw_rel), -math.cos(yaw_rel)

    def _mediana(self, valores):
        if not valores:
            return None
        vals = sorted(valores)
        n = len(vals)
        mid = n // 2
        if n % 2:
            return vals[mid]
        return 0.5 * (vals[mid - 1] + vals[mid])

    def _cluster_continuo(self, pts, along_fn):
        if not pts:
            return []
        orden = sorted(pts, key=along_fn)
        grupos = [[orden[0]]]
        for p in orden[1:]:
            if abs(along_fn(p) - along_fn(grupos[-1][-1])) <= 0.09:
                grupos[-1].append(p)
            else:
                grupos.append([p])

        def score(g):
            span = abs(along_fn(g[-1]) - along_fn(g[0])) if len(g) > 1 else 0.0
            return len(g) * max(0.05, span)

        return max(grupos, key=score)

    def _pared_axis_robusta(self, pts, dist_fn, along_fn, min_span=0.14):
        """Detecta una pared recta axis-aligned con consenso + tramo continuo.

        1. Agrupa por distancia perpendicular a la pared esperada.
        2. Refina la distancia con inliers cercanos.
        3. Descarta puntos aislados y usa solo el segmento continuo dominante.
        """
        if len(pts) < 8:
            return None, [], {'n': 0, 'span': 0.0, 'err': None}

        bin_w = 0.025
        tol = 0.040
        bins = {}
        for p in pts:
            d = dist_fn(p)
            if not math.isfinite(d):
                continue
            b = int(round(d / bin_w))
            bins.setdefault(b, []).append(p)

        mejor = None
        for b in bins:
            centro = b * bin_w
            cercanos = [p for p in pts if abs(dist_fn(p) - centro) <= tol]
            if len(cercanos) < 6:
                continue

            base = self._mediana([dist_fn(p) for p in cercanos])
            inliers = [p for p in cercanos if abs(dist_fn(p) - base) <= tol]
            cluster = self._cluster_continuo(inliers, along_fn)
            if len(cluster) < 5:
                continue

            along = sorted(along_fn(p) for p in cluster)
            span = along[-1] - along[0] if len(along) > 1 else 0.0
            if span < min_span:
                continue

            distancias = sorted(dist_fn(p) for p in cluster)
            k = max(0, int(len(distancias) * 0.15))
            recortadas = (distancias[k:len(distancias) - k]
                          if len(distancias) - 2 * k >= 4 else distancias)
            distancia = sum(recortadas) / len(recortadas)
            err = sum(abs(dist_fn(p) - distancia) for p in cluster) / len(cluster)
            puntaje = len(cluster) * span / (err + 0.01)
            cand = (puntaje, distancia, cluster, {'n': len(cluster),
                                                  'span': span,
                                                  'err': err})
            if mejor is None or cand[0] > mejor[0]:
                mejor = cand

        if mejor is None:
            return None, [], {'n': 0, 'span': 0.0, 'err': None}
        return mejor[1], mejor[2], mejor[3]

    def _dist_pared_izq_inicio(self):
        pts = [
            p for p in self._lidar_full
            if 0.05 <= p[1] <= 0.70 and -0.25 <= p[0] <= 0.85
        ]
        return self._pared_axis_robusta(pts, lambda p: p[1], lambda p: p[0],
                                        min_span=0.22)

    def _dist_pared_atras_inicio(self):
        pts = [
            p for p in self._lidar_full
            if -0.70 <= p[0] <= -0.05 and abs(p[1]) <= 0.45
        ]
        return self._pared_axis_robusta(pts, lambda p: -p[0], lambda p: p[1],
                                        min_span=0.12)

    def _detectar_paredes_lidar(self):
        detectadas = []
        segmentos = []
        regiones = [
            # pared izquierda
            ([p for p in self._lidar_full
              if 0.05 <= p[1] <= 1.00 and -0.35 <= p[0] <= 1.25],
             lambda p: p[1], lambda p: p[0], 0.16),
            # pared derecha
            ([p for p in self._lidar_full
              if -1.00 <= p[1] <= -0.05 and -0.35 <= p[0] <= 1.25],
             lambda p: -p[1], lambda p: p[0], 0.16),
            # pared frontal
            ([p for p in self._lidar_full
              if 0.05 <= p[0] <= 1.25 and -0.80 <= p[1] <= 0.80],
             lambda p: p[0], lambda p: p[1], 0.16),
            # pared trasera visible en esquinas
            ([p for p in self._lidar_full
              if -0.80 <= p[0] <= -0.05 and -0.80 <= p[1] <= 0.80],
             lambda p: -p[0], lambda p: p[1], 0.12),
        ]
        for pts, dist_fn, along_fn, min_span in regiones:
            _, inliers, _ = self._pared_axis_robusta(
                pts, dist_fn, along_fn, min_span=min_span)
            detectadas.extend(inliers)
            if len(inliers) >= 2:
                orden = sorted(inliers, key=along_fn)
                segmentos.append([[round(x, 3), round(y, 3)] for x, y in orden[:120]])

        # Deduplicar por centimetro para no mandar puntos repetidos.
        vistos = set()
        salida = []
        for x, y in detectadas:
            key = (round(x, 2), round(y, 2))
            if key in vistos:
                continue
            vistos.add(key)
            salida.append([round(x, 3), round(y, 3)])
        self._wall_pts = salida[:240]
        self._wall_segments = segmentos[:8]

    def _calibrar_inicio_con_lidar(self, yaw, factor_ang):
        """Ajuste suave de la pose inicial usando la esquina visible por LiDAR.

        Solo actua en el primer tramo: ancla la distancia a la pared izquierda
        del carril A y, si el LiDAR ve la pared inferior, ancla tambien Y.
        Despues la posicion queda a cargo de avance + rumbo/giros.
        """
        if self._odom_origen is None or self.odom_cm / 100.0 > self.track_lidar_init_m:
            self._calib_info['activa'] = False
            return

        yaw_rel = abs(self._yaw_rel_pista(yaw, factor_ang))
        if yaw_rel > math.radians(35.0):
            return

        a = max(0.0, min(1.0, self.track_lidar_blend))
        d_izq, pts_izq, info_izq = self._dist_pared_izq_inicio()
        d_atras, pts_atras, info_atras = self._dist_pared_atras_inicio()

        if d_izq is not None:
            objetivo_x = max(0.12, min(0.88, d_izq / CELDA_M))
            self._track_pos[0] = (1.0 - a) * self._track_pos[0] + a * objetivo_x

        if d_atras is not None and self.odom_cm < 25.0:
            # Centro del robot en A7 (dibuja en fila 6.5): la pared trasera es
            # el borde inferior (y=8), a ~1.5 celdas atrás.
            objetivo_y = max(6.05, min(6.95, 8.0 - d_atras / CELDA_M))
            self._track_pos[1] = (1.0 - a) * self._track_pos[1] + a * objetivo_y

        if self.odom_cm < 25.0 and self.traj:
            p0 = [round(self._track_pos[0], 3), round(self._track_pos[1], 3)]
            self.traj[0] = p0
            self._ultimo_tray = (self._track_pos[0], self._track_pos[1])

        self._calib_info = {
            'activa': True,
            'izq_m': round(d_izq, 3) if d_izq is not None else None,
            'atras_m': round(d_atras, 3) if d_atras is not None else None,
            'izq_pts': info_izq['n'],
            'atras_pts': info_atras['n'],
            'izq_span': round(info_izq['span'], 3),
            'atras_span': round(info_atras['span'], 3),
        }
        pts = pts_izq + pts_atras
        self._calib_pts = [[round(x, 3), round(y, 3)] for x, y in pts]

    # ── Callbacks ───────────────────────────────────────────────────────
    def cb_scan(self, msg: LaserScan):
        pts = []
        pts_full = []
        n = len(msg.ranges)
        paso = max(1, n // 360)         # decimado: como mucho ~360 puntos
        for i in range(0, n, paso):
            r = msg.ranges[i]
            if not math.isfinite(r) or r < msg.range_min or r > 3.5:
                continue
            af = _norm(msg.angle_min + i * msg.angle_increment - self.front_rad)
            x = r * math.cos(af)
            y = r * math.sin(af)
            if r <= 1.4:
                pts_full.append((x, y))
            if abs(af) > self.lidar_fov_rad / 2.0:
                continue
            pts.append([round(x, 3), round(y, 3)])
        with self.lock:
            self.lidar = pts
            self._lidar_full = pts_full
            self._detectar_paredes_lidar()
            frente = [x for x, y in pts_full if x > 0.03 and abs(y) <= 0.12]
            atras = [-x for x, y in pts_full if x < -0.03 and abs(y) <= 0.12]
            izq = [y for x, y in pts_full if y > 0.03 and abs(x) <= 0.12]
            der = [-y for x, y in pts_full if y < -0.03 and abs(x) <= 0.12]
            if frente:
                self.d_frente = round(min(frente), 3)
            if atras:
                self.d_atras = round(min(atras), 3)
            if izq:
                self.d_izq = round(min(izq), 3)
            if der:
                self.d_der = round(min(der), 3)

    def cb_imu(self, msg: Imu):
        """Yaw del IMU: rumbo preciso para el dibujo (sin patinaje de rueda).

        Usa la orientación fusionada si el IMU la entrega; si solo publica
        velocidad angular (orientación degenerada, REP-145), integra
        angular_velocity.z. El dibujo trabaja en yaw RELATIVO al arranque."""
        q = msg.orientation
        if self.imu_fuente == 'gyro':
            usar_gyro = True
        elif self.imu_fuente == 'orientacion':
            usar_gyro = False
        else:   # 'auto'
            usar_gyro = (msg.orientation_covariance[0] < 0.0 or
                         (q.x == 0.0 and q.y == 0.0 and
                          q.z == 0.0 and q.w == 0.0))
        if usar_gyro:
            now = self.get_clock().now()
            with self.lock:
                if self._t_imu_prev is not None:
                    dt = (now - self._t_imu_prev).nanoseconds * 1e-9
                    if 0.0 < dt < 0.5:
                        self._imu_yaw_int += msg.angular_velocity.z * dt
                self._t_imu_prev = now
                self.imu_yaw = self._imu_yaw_int
        else:
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            with self.lock:
                self.imu_yaw = yaw

    def cb_camera(self, msg: Image):
        now_s = self.get_clock().now().nanoseconds * 1e-9
        with self.lock:
            if self._camera_period > 0.0 and now_s - self._camera_last_emit_s < self._camera_period:
                return
            self._camera_last_emit_s = now_s

        src = _imagen_a_jpeg_data_url(
            msg, self.camera_max_width, self.camera_jpeg_quality)
        info = {
            'ok': src is not None,
            'topic': self.topic_camera,
            'encoding': msg.encoding,
            'w': int(msg.width),
            'h': int(msg.height),
        }
        with self.lock:
            self.camera_info = info
            if src is not None:
                self.camera_src = src

    def cb_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # yaw de ODOM: SOLO para saber si el paso es hacia adelante o atrás
        # (el desplazamiento dx,dy está en el marco de la odometría).
        odom_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self.lock:
            if self.fuente_yaw == 'imu':
                if self.imu_yaw is None:
                    return
                yaw = self.imu_yaw
                factor_ang = self.factor_ang_imu
            else:
                yaw = odom_yaw
                factor_ang = self.factor_ang_odom

            if self._odom_origen is None:
                self._odom_origen = (p.x, p.y, yaw)
                self._ultimo_odom = (p.x, p.y)
                self._ultimo_yaw = odom_yaw
                self.pos = [round(self._track_pos[0], 3), round(self._track_pos[1], 3)]
                hx, hy = self._rumbo_desde_rel(0.0, snap=False)
                self.head = [round(hx, 3), round(hy, 3)]
                self.traj.append(list(self.pos))
                self._ultimo_tray = (self._track_pos[0], self._track_pos[1])
                return

            if self._ultimo_odom is not None:
                dx = p.x - self._ultimo_odom[0]
                dy = p.y - self._ultimo_odom[1]
                paso_m = math.hypot(dx, dy) * self.factor_dist_odom
                if paso_m > 0.0:
                    # ¿Retrocede? Se mide en el marco ODOM (dx,dy vs rumbo odom).
                    fwd_x = math.cos(self._ultimo_yaw if self._ultimo_yaw is not None else odom_yaw)
                    fwd_y = math.sin(self._ultimo_yaw if self._ultimo_yaw is not None else odom_yaw)
                    if dx * fwd_x + dy * fwd_y < -0.002:
                        paso_m = -paso_m

                    # Dirección del paso sobre la pista = yaw configurado.
                    yaw_rel = self._yaw_rel_pista(yaw, factor_ang)
                    hx_mov, hy_mov = self._rumbo_desde_rel(
                        yaw_rel, snap=self.track_snap_heading)
                    self._track_pos[0] += hx_mov * (paso_m / CELDA_M)
                    self._track_pos[1] += hy_mov * (paso_m / CELDA_M)
                    self.odom_cm += abs(paso_m) * 100.0

            self._calibrar_inicio_con_lidar(yaw, factor_ang)

            self._ultimo_odom = (p.x, p.y)
            self._ultimo_yaw = odom_yaw
            px, py = self._track_pos
            self.pos = [round(px, 3), round(py, 3)]
            hx, hy = self._rumbo_desde_rel(
                self._yaw_rel_pista(yaw, factor_ang), snap=False)
            self.head = [round(hx, 3), round(hy, 3)]
            if (self._ultimo_tray is None or
                    math.hypot(px - self._ultimo_tray[0],
                               py - self._ultimo_tray[1]) * CELDA_M >= TRAY_PASO_MIN_M):
                self.traj.append([round(px, 3), round(py, 3)])
                self._ultimo_tray = (px, py)

    def cb_pare(self, msg: Bool):
        with self.lock:
            # Registrar solo el flanco False->True. Antes se agregaba un punto
            # por frame y una detección producía una franja roja en el mapa.
            if msg.data and not self._pare_activo:
                if self.pos is not None:
                    self.pare.append(list(self.pos))
            self._pare_activo = bool(msg.data)

    def cb_verde(self, msg: Bool):
        with self.lock:
            en_zona_meta = False
            if self.pos is not None:
                columna = math.floor(self.pos[0])
                fila = math.floor(self.pos[1])
                en_zona_meta = (
                    columna in META_COLUMNAS_VALIDAS and
                    fila in META_FILAS_VALIDAS)

            # Fuera de J1–L2 el verde se trata como no detectado: no aparece
            # rectángulo en cámara ni marcador en el diagrama.
            verde_valido = bool(msg.data) and en_zona_meta

            # Solo existe un marcador META durante todo el recorrido, aunque
            # la detección parpadee o la cámara vuelva a verla más adelante.
            if (verde_valido and not self._verde_activo and
                    not self._meta_registrada and self.pos is not None):
                self.verde_pos.append(list(self.pos))
                self._meta_registrada = True
            self.verde_detectado = verde_valido
            self._verde_activo = verde_valido

    def cb_estado(self, msg: String):
        with self.lock:
            # Un tramo nuevo comienza al volver al avance tras un giro.
            if msg.data == 'AVANZAR_PARALELO' and self.estado in ('GIRAR', 'AVANCE_GIRO_VACIO'):
                self.tramo += 1
            self.estado = msg.data

    def cb_metricas(self, msg: String):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            df = d.get('d_frente')
            da = d.get('d_atras')
            di = d.get('d_izq')
            dd = d.get('d_der')
            dlf = d.get('d_lado_frontal')
            dlt = d.get('d_lado_trasera')
            v_cmd = d.get('v')
            w_cmd = d.get('w')
            self.d_frente = df if (df is not None and math.isfinite(df)) else self.d_frente
            self.d_atras = da if (da is not None and math.isfinite(da)) else self.d_atras
            self.d_izq = di if (di is not None and math.isfinite(di)) else self.d_izq
            self.d_der = dd if (dd is not None and math.isfinite(dd)) else self.d_der
            self.d_lado_frontal = dlf if (dlf is not None and math.isfinite(dlf)) else self.d_lado_frontal
            self.d_lado_trasera = dlt if (dlt is not None and math.isfinite(dlt)) else self.d_lado_trasera
            self.v_cmd = float(v_cmd) if v_cmd is not None and math.isfinite(v_cmd) else self.v_cmd
            self.w_cmd = float(w_cmd) if w_cmd is not None and math.isfinite(w_cmd) else self.w_cmd
            giros = d.get('giros_fisicos')
            if giros is not None:
                self.giros_fisicos = int(giros)

    def cb_ruta(self, msg: String):
        """Ruta corta calculada por maze_solver (fase 2). Solo se reenvia por
        HTTP para pintarla de amarillo; el visualizador no controla nada.

        Convierte cada celda (col,row) a su CENTRO en coords de pista
        (col+0.5, row+0.5), misma convencion que pos/traj (A7 -> 0.5,6.5)."""
        try:
            d = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        pts = []
        for cr in d.get('celdas') or []:
            try:
                c = float(cr[0])
                r = float(cr[1])
            except (TypeError, IndexError, ValueError):
                continue
            pts.append([round(c + 0.5, 3), round(r + 0.5, 3)])
        with self.lock:
            self.ruta_corta = pts

    # ── Snapshot JSON para el frontend ──────────────────────────────────
    def snapshot(self):
        with self.lock:
            if self.estado == 'AVANZAR_PARALELO':
                movimiento = 'SIGUIENDO PARED IZQUIERDA'
            elif self.estado == 'PAUSA_CHEQUEO_PARED':
                movimiento = 'DETENIDO: VERIFICANDO LADO IZQUIERDO'
            elif self.estado == 'AVANCE_GIRO_VACIO':
                movimiento = 'AVANCE RECTO ENTRE LOS DOS GIROS'
            elif self.estado == 'ALINEAR':
                movimiento = 'ALINEANDO CON LA PARED'
            elif self.estado == 'META':
                movimiento = 'META VERDE: DETENIDO'
            elif self.estado == 'ESPERA_RUTA':
                movimiento = 'RUTA LISTA (AMARILLA): DETENIDO, ESPERANDO PARTIR'
            elif self.estado == 'SEGUIR_RUTA':
                movimiento = 'RUTA CORTA: SIGUIENDO RUTA AMARILLA'
            elif self.estado == 'FRENO_PARE':
                movimiento = 'PARE ROJO: FRENO DURANTE 5 SEGUNDOS'
            elif self.estado == 'DETENIDO':
                movimiento = 'DETENIDO POR SEGURIDAD'
            elif self.estado == 'CORREGIR_GIRO':
                movimiento = 'CORRIGIENDO (giró >360°): reorientando'
            elif self.estado == 'GIRAR':
                if self.w_cmd > 0.02:
                    movimiento = 'GIRANDO A LA IZQUIERDA'
                elif self.w_cmd < -0.02:
                    movimiento = 'GIRANDO A LA DERECHA'
                else:
                    movimiento = 'PREPARANDO GIRO'
            else:
                movimiento = self.estado
            return {
                'estado': self.estado, 'tramo': self.tramo,
                'giros_fisicos': self.giros_fisicos,
                'odom_cm': round(self.odom_cm, 1),
                'lidar': self.lidar,
                'd_frente': self.d_frente,
                'd_atras': self.d_atras,
                'd_izq': self.d_izq, 'd_der': self.d_der,
                'd_lado_frontal': self.d_lado_frontal,
                'd_lado_trasera': self.d_lado_trasera,
                'v_cmd': round(self.v_cmd, 3),
                'w_cmd': round(self.w_cmd, 3),
                'movimiento': movimiento,
                'traj': list(self.traj), 'pos': self.pos, 'head': self.head,
                'pare': list(self.pare), 'n_pts': len(self.traj),
                'verde_detectado': self.verde_detectado,
                'verde_pos': list(self.verde_pos),
                'ruta_corta': list(self.ruta_corta),
                'calib': dict(self._calib_info),
                'wall_pts': list(self._wall_pts),
                'wall_segments': list(self._wall_segments),
                'calib_pts': list(self._calib_pts),
                'camera': dict(self.camera_info, src=self.camera_src),
            }

    # ── Servidor HTTP (solo /data, con CORS) ────────────────────────────
    def _arrancar_http(self):
        nodo = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass    # silenciar el log de cada request

            def _cors(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'no-store')

            def do_GET(self):
                if self.path.startswith('/data'):
                    cuerpo = json.dumps(nodo.snapshot()).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self._cors()
                    self.send_header('Content-Length', str(len(cuerpo)))
                    self.end_headers()
                    self.wfile.write(cuerpo)
                else:
                    # ayuda mínima si abren la IP directo en el navegador
                    cuerpo = (b'visualizador_web activo. El frontend esta en la '
                              b'laptop (web/index.html). Datos en /data')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain; charset=utf-8')
                    self._cors()
                    self.send_header('Content-Length', str(len(cuerpo)))
                    self.end_headers()
                    self.wfile.write(cuerpo)

        self._srv = ThreadingHTTPServer(('0.0.0.0', self.puerto), Handler)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()


def main(args=None):
    rclpy.init(args=args)
    nodo = VisualizadorWeb()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
