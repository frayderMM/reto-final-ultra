#!/usr/bin/env python3
"""
pare_detector.py — Detección de la señal PARE por cámara (Gran Prix).

REUTILIZA el pipeline del detector de señales del RC de visión
(Line_Detector_RosCar_Pi_R2 / lane_detector.py): segmentación por color
en HSV + morfología + filtrado de blobs por forma con
connectedComponentsWithStats. Cambios respecto al RC:

  · El color objetivo es el ROJO del cartel PARE. El rojo en HSV cruza
    el 0 del hue, así que se combinan DOS rangos (0–h_bajo y h_alto–179)
    en vez del rango único del amarillo.
  · El filtro de forma es el inverso del RC: allí se buscaban cintas
    ALARGADAS (elongación alta); un cartel PARE es un blob COMPACTO
    (relación de aspecto ~1, solidez alta). Un reflejo rojo alargado o
    un cable rojo NO pasan el filtro.
  · Zona de atención (fusión por contexto): maze_solver publica
    /maze/atencion_pare=True cuando el LiDAR indica que se aproxima una
    intersección. Fuera de la zona de atención se exige un área mínima
    mayor (factor_area_sin_atencion) — el rojo lejano/espurio de mitad
    de pasillo no dispara nada.
  · Banda central: el PARE solo cuenta si su centro cae en la BANDA
    CENTRAL horizontal del frame (centro_tol_frac). Un cartel a un lado
    (que el robot no está mirando de frente) se ignora.

Tópicos:
    sub  /image_raw            (sensor_msgs/Image)
    sub  /maze/atencion_pare   (std_msgs/Bool)
    pub  /pare_detectado       (std_msgs/Bool)   — ROJO (PARE) confirmado, centrado
    pub  /verde_detectado      (std_msgs/Bool)   — VERDE opaco confirmado, centrado
    pub  /amarillo_detectado   (std_msgs/Bool)   — AMARILLO confirmado
    pub  /beep                 (std_msgs/UInt16) — secuencia "pi pi piiiiiii" al ver rojo/amarillo
    pub  /pare/area            (std_msgs/Float32) — área del blob rojo (debug/tuning)
    pub  /pare/debug_image     (sensor_msgs/Image) — overlay rojo+verde para captura
"""

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, UInt16
from cv_bridge import CvBridge


class PareDetector(Node):
    def __init__(self):
        super().__init__('pare_detector')
        self.bridge = CvBridge()

        self.declare_parameters('', [
            # Rojo en HSV — DOS rangos porque el hue del rojo cruza 0/179
            ('rojo_h_bajo_max',   15),
            ('rojo_h_alto_min',  165),
            ('rojo_s_min',        80),
            ('rojo_v_min',        50),
            # Amarillo: rango HSV separado del verde y del rojo.
            ('amarillo_h_min',    20),
            ('amarillo_h_max',    35),
            ('amarillo_s_min',   100),
            ('amarillo_v_min',    80),
            ('amarillo_area_min', 600),
            # Verde OPACO/apagado (cartel META, verde sage poco saturado):
            # s_min bajo para agarrar el verde grisáceo; h fuera del beige de la
            # pared (naranja) y el blanco/gris queda fuera por S baja.
            ('verde_h_min',       35),
            ('verde_h_max',       95),
            ('verde_s_min',       40),
            ('verde_s_max',      255),
            ('verde_v_min',       60),
            ('verde_v_max',      255),
            # Descarta motas/reflejos, pero conserva carteles, tubos y objetos.
            ('verde_area_min',    600),
            ('verde_aspecto_min', 0.05),
            ('verde_aspecto_max', 20.0),
            ('verde_area_max',  150000),
            ('verde_solidez_min', 0.35),
            # Forma del cartel (blob compacto, no cinta)
            ('area_min',         200),
            ('area_max',       60000),
            ('aspecto_min',      0.5),   # w/h del bounding box
            ('aspecto_max',      2.0),
            ('solidez_min',      0.75),  # area / area del convex hull
            # Fraccion vertical (desde arriba) donde se busca el ROJO (y el
            # amarillo). 1.0 = TODA la camara.
            ('franja_inferior',  1.0),
            # El VERDE (meta) se busca en los 2/3 SUPERIORES de la imagen
            # (0.6667). Todo lo del tercio inferior queda fuera.
            ('franja_verde',     0.6667),
            # Solo se acepta el PARE si su centro está en la BANDA CENTRAL
            # horizontal: |cx_blob - cx_img| <= centro_tol_frac * ancho.
            ('centro_tol_frac',  0.20),  # 0.20 → banda central del 40% del ancho
            # Fusión por contexto (zona de atención del maze_solver)
            ('factor_area_sin_atencion', 2.5),
            ('frames_confirmacion', 3),  # frames seguidos para rechazar destellos
            ('publish_debug',    True),
            ('topic_camera',      '/image_raw'),
        ])

        gp = lambda n: self.get_parameter(n).value
        self.h_bajo = int(gp('rojo_h_bajo_max'))
        self.h_alto = int(gp('rojo_h_alto_min'))
        self.s_min = int(gp('rojo_s_min'))
        self.v_min = int(gp('rojo_v_min'))
        self.amarillo_h_min = int(gp('amarillo_h_min'))
        self.amarillo_h_max = int(gp('amarillo_h_max'))
        self.amarillo_s_min = int(gp('amarillo_s_min'))
        self.amarillo_v_min = int(gp('amarillo_v_min'))
        self.amarillo_area_min = float(gp('amarillo_area_min'))
        self.verde_h_min = int(gp('verde_h_min'))
        self.verde_h_max = int(gp('verde_h_max'))
        self.verde_s_min = int(gp('verde_s_min'))
        self.verde_s_max = int(gp('verde_s_max'))
        self.verde_v_min = int(gp('verde_v_min'))
        self.verde_v_max = int(gp('verde_v_max'))
        self.verde_area_min = float(gp('verde_area_min'))
        self.verde_aspecto_min = float(gp('verde_aspecto_min'))
        self.verde_aspecto_max = float(gp('verde_aspecto_max'))
        self.verde_area_max = float(gp('verde_area_max'))
        self.verde_solidez_min = float(gp('verde_solidez_min'))
        self.area_min = float(gp('area_min'))
        self.area_max = float(gp('area_max'))
        self.aspecto_min = float(gp('aspecto_min'))
        self.aspecto_max = float(gp('aspecto_max'))
        self.solidez_min = float(gp('solidez_min'))
        self.franja_inferior = float(gp('franja_inferior'))
        self.franja_verde = float(gp('franja_verde'))
        self.centro_tol_frac = float(gp('centro_tol_frac'))
        self.factor_sin_atencion = float(gp('factor_area_sin_atencion'))
        self.frames_confirmacion = int(gp('frames_confirmacion'))
        self.publish_debug = bool(gp('publish_debug'))
        self.topic_camera = str(gp('topic_camera'))

        self.atencion = False
        self.frames_seguidos = 0
        self.frames_seguidos_verde = 0
        self.frames_seguidos_amarillo = 0
        self._rojo_anterior = False
        self._amarillo_anterior = False
        self._beep_timer = None
        self._beep_paso = 0

        self.create_subscription(Image, self.topic_camera, self.on_image, 10)
        self.create_subscription(Bool, '/maze/atencion_pare', self.on_atencion, 10)
        self.pub_pare = self.create_publisher(Bool, '/pare_detectado', 10)
        self.pub_verde = self.create_publisher(Bool, '/verde_detectado', 10)
        self.pub_amarillo = self.create_publisher(Bool, '/amarillo_detectado', 10)
        self.pub_beep = self.create_publisher(UInt16, '/beep', 10)
        self.pub_area = self.create_publisher(Float32, '/pare/area', 10)
        self.pub_dbg = self.create_publisher(Image, '/pare/debug_image', 10)

        self.get_logger().info(
            f'pare_detector listo en {self.topic_camera} | '
            f'rojo H<= {self.h_bajo} o H>={self.h_alto}, '
            f'S>={self.s_min}, V>={self.v_min} | area>={self.area_min}')

    def on_atencion(self, msg: Bool):
        self.atencion = msg.data

    # ------------------------------------------------------------------
    def on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        h, w = frame.shape[:2]
        # ROJO y AMARILLO: toda la cámara (franja_inferior=1.0).
        hsv = cv2.cvtColor(frame[:int(self.franja_inferior * h), :],
                           cv2.COLOR_BGR2HSV)
        # VERDE (meta): SOLO la banda de arriba (franja_verde=1/3).
        hsv_verde = cv2.cvtColor(frame[:max(1, int(self.franja_verde * h)), :],
                                 cv2.COLOR_BGR2HSV)

        # área mínima según contexto: fuera de la zona de atención se exige
        # un cartel más grande (más cerca) para aceptarlo
        area_min = self.area_min if self.atencion \
            else self.area_min * self.factor_sin_atencion

        # ROJO (PARE): en TODA la cámara y en CUALQUIER posición (no exige
        # que el cartel esté centrado -> exigir_centro=False).
        mask_rojo = self._mascara_rojo(hsv)
        mask_rojo = self._quitar_componentes_pequenos(mask_rojo, area_min)
        det_r, mejor_r = self._validar_forma(mask_rojo, area_min,
                                             exigir_centro=False)
        self.frames_seguidos = self.frames_seguidos + 1 if det_r else 0
        conf_r = self.frames_seguidos >= self.frames_confirmacion

        # VERDE (opaco) -- solo en la banda superior (hsv_verde)
        mask_verde = self._mascara_verde(hsv_verde)
        mask_verde = self._quitar_componentes_pequenos(
            mask_verde, self.verde_area_min)
        det_v, mejor_v = self._validar_forma(
            mask_verde, self.verde_area_min,
            aspecto_min=self.verde_aspecto_min,
            aspecto_max=self.verde_aspecto_max,
            area_max=self.verde_area_max,
            solidez_min=self.verde_solidez_min,
            exigir_centro=False)
        self.frames_seguidos_verde = self.frames_seguidos_verde + 1 if det_v else 0
        conf_v = self.frames_seguidos_verde >= self.frames_confirmacion

        # AMARILLO. Usa el mismo filtro compacto del cartel rojo y exige
        # confirmación temporal para no activar el buzzer por reflejos.
        mask_amarillo = self._mascara_amarillo(hsv)
        mask_amarillo = self._quitar_componentes_pequenos(
            mask_amarillo, self.amarillo_area_min)
        det_a, mejor_a = self._validar_forma(
            mask_amarillo, self.amarillo_area_min)
        self.frames_seguidos_amarillo = (
            self.frames_seguidos_amarillo + 1 if det_a else 0)
        conf_a = self.frames_seguidos_amarillo >= self.frames_confirmacion

        self.pub_pare.publish(Bool(data=bool(conf_r)))
        self.pub_verde.publish(Bool(data=bool(conf_v)))
        self.pub_amarillo.publish(Bool(data=bool(conf_a)))
        self.pub_area.publish(Float32(
            data=float(mejor_r['area']) if mejor_r else 0.0))

        # Secuencia "pi pi piiiiiii" (ver _disparar_beep_paso). Se dispara
        # una sola vez al comenzar cada detección, no por frame.
        if ((conf_r and not self._rojo_anterior) or
                (conf_a and not self._amarillo_anterior)):
            self._iniciar_beep()
        self._rojo_anterior = conf_r
        self._amarillo_anterior = conf_a

        if self.publish_debug:
            self._publicar_debug(frame, mask_rojo, mask_verde, mask_amarillo,
                                 mejor_r, mejor_v, mejor_a,
                                 conf_r, conf_v, conf_a, msg)

    # ------------------------------------------------------------------
    def _limpiar(self, mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _mascara_rojo(self, hsv):
        # el hue del rojo cruza 0/179 → dos rangos
        lo1 = np.array([0, self.s_min, self.v_min], dtype=np.uint8)
        hi1 = np.array([self.h_bajo, 255, 255], dtype=np.uint8)
        lo2 = np.array([self.h_alto, self.s_min, self.v_min], dtype=np.uint8)
        hi2 = np.array([179, 255, 255], dtype=np.uint8)
        return self._limpiar(cv2.bitwise_or(cv2.inRange(hsv, lo1, hi1),
                                            cv2.inRange(hsv, lo2, hi2)))

    def _mascara_verde(self, hsv):
        lo = np.array([self.verde_h_min, self.verde_s_min, self.verde_v_min], dtype=np.uint8)
        hi = np.array([self.verde_h_max, self.verde_s_max, self.verde_v_max], dtype=np.uint8)
        return self._limpiar(cv2.inRange(hsv, lo, hi))

    def _mascara_amarillo(self, hsv):
        lo = np.array([self.amarillo_h_min, self.amarillo_s_min,
                       self.amarillo_v_min], dtype=np.uint8)
        hi = np.array([self.amarillo_h_max, 255, 255], dtype=np.uint8)
        return self._limpiar(cv2.inRange(hsv, lo, hi))

    # "pi pi piiiiiii": dos pitidos cortos + uno largo. Cada valor es la
    # duracion (ms) de un one-shot del buzzer (se apaga solo); la pausa
    # entre pasos es mayor que el pitido corto para que quede un silencio
    # audible entre "pi" y "pi" antes del pitido largo final.
    _BEEP_DURACIONES_MS = (120, 120, 900)
    _BEEP_PAUSA_S = 0.27

    def _iniciar_beep(self):
        """Dispara la secuencia sin bloquear el procesamiento de cámara."""
        self._beep_paso = 0
        self._disparar_beep_paso()

    def _disparar_beep_paso(self):
        self.pub_beep.publish(UInt16(data=self._BEEP_DURACIONES_MS[self._beep_paso]))
        if self._beep_timer is not None:
            self._beep_timer.cancel()
            self.destroy_timer(self._beep_timer)
            self._beep_timer = None
        self._beep_paso += 1
        if self._beep_paso < len(self._BEEP_DURACIONES_MS):
            self._beep_timer = self.create_timer(
                self._BEEP_PAUSA_S, self._disparar_beep_paso)

    @staticmethod
    def _quitar_componentes_pequenos(mask, area_min):
        """Conserva solo regiones conectadas suficientemente grandes.

        Se aplica antes del overlay para que la web no pinte motas verdes que
        nunca podrían convertirse en una detección válida.
        """
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        limpia = np.zeros_like(mask)
        for etiqueta in range(1, n):
            if stats[etiqueta, cv2.CC_STAT_AREA] >= area_min:
                limpia[labels == etiqueta] = 255
        return limpia

    # ------------------------------------------------------------------
    def _validar_forma(self, mask, area_min, aspecto_min=None,
                       aspecto_max=None, area_max=None, solidez_min=None,
                       exigir_centro=True):
        """Valida que el blob rojo tenga forma de CARTEL: compacto
        (aspecto ~1) y sólido (sin huecos grandes) — un reflejo alargado
        o ruido disperso no pasa. Además solo lo acepta si está en la BANDA
        CENTRAL horizontal de la imagen (si el PARE no está al centro, no
        cuenta). Es la validación por contorno que pide el enunciado."""
        aspecto_min = self.aspecto_min if aspecto_min is None else aspecto_min
        aspecto_max = self.aspecto_max if aspecto_max is None else aspecto_max
        area_max = self.area_max if area_max is None else area_max
        solidez_min = self.solidez_min if solidez_min is None else solidez_min
        cx_img = mask.shape[1] / 2.0
        tol_px = self.centro_tol_frac * mask.shape[1]
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        mejor = None
        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area < area_min or area > area_max:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if exigir_centro and abs((x + bw / 2.0) - cx_img) > tol_px:
                continue
            aspecto = bw / float(bh)
            if not (aspecto_min <= aspecto <= aspecto_max):
                continue
            hull = cv2.convexHull(cnt)
            area_hull = cv2.contourArea(hull)
            if area_hull < 1e-3 or area / area_hull < solidez_min:
                continue
            if mejor is None or area > mejor['area']:
                mejor = {'area': area, 'bbox': (x, y, bw, bh)}
        return mejor is not None, mejor

    # ------------------------------------------------------------------
    def _publicar_debug(self, frame, mask_rojo, mask_verde, mask_amarillo,
                        mejor_r, mejor_v, mejor_a,
                        conf_r, conf_v, conf_a, header_msg):
        dbg = frame.copy()
        overlay = dbg.copy()
        # Cada máscara puede tener distinto alto (rojo/amarillo = todo el frame,
        # verde = solo la banda de arriba): pintar cada una con su propio alto.
        overlay[:mask_rojo.shape[0]][mask_rojo > 0] = (0, 0, 255)        # rojo
        overlay[:mask_verde.shape[0]][mask_verde > 0] = (0, 200, 0)      # verde
        overlay[:mask_amarillo.shape[0]][mask_amarillo > 0] = (0, 220, 255)  # amarillo
        cv2.addWeighted(overlay, 0.4, dbg, 0.6, 0, dbg)
        Hf, Wf = dbg.shape[:2]
        # El ROJO se acepta en TODA la cámara (cualquier posición). El VERDE
        # solo por encima de esta línea (2/3 superiores).
        yv = int(self.franja_verde * Hf)
        cv2.line(dbg, (0, yv), (Wf, yv), (0, 200, 0), 1)
        cv2.putText(dbg, 'VERDE arriba de esta linea', (5, max(14, yv - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        for mejor, conf, txt, col in ((mejor_r, conf_r, 'PARE', (0, 0, 255)),
                                      (mejor_v, conf_v, 'VERDE', (0, 200, 0)),
                                      (mejor_a, conf_a, 'AMARILLO', (0, 220, 255))):
            if mejor is None:
                continue
            x, y, bw, bh = mejor['bbox']
            color = (0, 255, 0) if conf else col
            cv2.rectangle(dbg, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(dbg, f'{txt} {mejor["area"]:.0f}px', (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        etq = []
        if conf_r:
            etq.append('PARE!')
        if conf_v:
            etq.append('VERDE!')
        if conf_a:
            etq.append('AMARILLO!')
        estado = ' '.join(etq) if etq else ('atencion' if self.atencion else '-')
        cv2.putText(dbg, estado, (5, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        out = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
        out.header = header_msg.header
        self.pub_dbg.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PareDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
