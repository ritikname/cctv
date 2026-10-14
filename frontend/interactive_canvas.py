"""
Interactive Video Canvas Widget for PySide6

Features:
- Maintains aspect-ratio-scaled video display.
- Interactive, draggable vertical & horizontal lines for Active Tracking Area (ROI).
  - Shaded mask outside the active tracking area so the user clearly sees the tracked region.
- Interactive, draggable Entry Gate Line (Tripwire) with draggable endpoints & body.
  - Perpendicular direction indicators (IN / OUT).
- Normalised coordinate representation [0.0, 1.0] for resolution independence.
"""

import math
from typing import Optional, Tuple
import numpy as np

from PySide6.QtCore import Qt, QRect, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPixmap, QImage,
    QCursor, QPolygonF
)
from PySide6.QtWidgets import QWidget


class InteractiveVideoCanvas(QWidget):
    roi_changed = Signal(float, float, float, float)  # left, top, right, bottom (norm)
    gate_line_changed = Signal(float, float, float, float)  # x1, y1, x2, y2 (norm)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(640, 420)
        self.setStyleSheet("background-color: #121212;")

        self._pixmap: Optional[QPixmap] = None
        self._frame_size: Tuple[int, int] = (1920, 1080)

        # ── Normalised ROI [0.0, 1.0]: (left, top, right, bottom) ─────────
        self.roi_left: float = 0.05
        self.roi_top: float = 0.05
        self.roi_right: float = 0.95
        self.roi_bottom: float = 0.95
        self.roi_enabled: bool = True

        # ── Normalised Gate Line [0.0, 1.0]: point_a -> point_b ──────────
        self.gate_p1: Tuple[float, float] = (0.20, 0.75)
        self.gate_p2: Tuple[float, float] = (0.80, 0.75)
        self.gate_direction: str = "below"  # "below" or "above" is IN
        self.gate_enabled: bool = True

        # Interaction state
        self._drag_target: Optional[str] = None
        self._drag_start_pos: Optional[QPointF] = None
        self._drag_initial_values = {}

        self.show_guides_during_play: bool = True

        # Live overlay counters
        self.live_in_count: int = 0
        self.live_out_count: int = 0
        self.live_statues_ignored: int = 0
        self.live_group_count: int = 0

    # ── Frame Updates ────────────────────────────────────────────────────────
    def set_frame(self, frame_or_qimg):
        """Accepts either a numpy BGR/RGB array or a QImage."""
        if frame_or_qimg is None:
            return

        if isinstance(frame_or_qimg, np.ndarray):
            h, w = frame_or_qimg.shape[:2]
            self._frame_size = (w, h)
            # Check if 3-channel
            if len(frame_or_qimg.shape) == 3:
                # Assume BGR from OpenCV
                import cv2
                rgb = cv2.cvtColor(frame_or_qimg, cv2.COLOR_BGR2RGB)
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            else:
                qimg = QImage(frame_or_qimg.data, w, h, w, QImage.Format_Grayscale8).copy()
            self._pixmap = QPixmap.fromImage(qimg)
        elif isinstance(frame_or_qimg, QImage):
            self._frame_size = (frame_or_qimg.width(), frame_or_qimg.height())
            self._pixmap = QPixmap.fromImage(frame_or_qimg)

        self.update()

    def set_counts(self, in_count: int, out_count: int, statues: int = 0, groups: int = 0):
        self.live_in_count = in_count
        self.live_out_count = out_count
        self.live_statues_ignored = statues
        self.live_group_count = groups
        self.update()

    def reset_geometry(self):
        self.roi_left = 0.05
        self.roi_top = 0.05
        self.roi_right = 0.95
        self.roi_bottom = 0.95
        self.gate_p1 = (0.20, 0.75)
        self.gate_p2 = (0.80, 0.75)
        self.update()
        self.roi_changed.emit(self.roi_left, self.roi_top, self.roi_right, self.roi_bottom)
        self.gate_line_changed.emit(self.gate_p1[0], self.gate_p1[1], self.gate_p2[0], self.gate_p2[1])

    def flip_gate_direction(self):
        self.gate_direction = "above" if self.gate_direction == "below" else "below"
        self.update()
        self.gate_line_changed.emit(self.gate_p1[0], self.gate_p1[1], self.gate_p2[0], self.gate_p2[1])

    # ── Coordinate Conversions ────────────────────────────────────────────────
    def _video_rect(self) -> QRectF:
        """Calculate display rectangle of the video with letterboxing."""
        ww = self.width()
        wh = self.height()
        fw, fh = self._frame_size
        if fw <= 0 or fh <= 0:
            return QRectF(0, 0, ww, wh)

        scale = min(ww / fw, wh / fh)
        disp_w = fw * scale
        disp_h = fh * scale
        ox = (ww - disp_w) / 2.0
        oy = (wh - disp_h) / 2.0
        return QRectF(ox, oy, disp_w, disp_h)

    def _norm_to_widget(self, nx: float, ny: float) -> QPointF:
        vr = self._video_rect()
        wx = vr.x() + nx * vr.width()
        wy = vr.y() + ny * vr.height()
        return QPointF(wx, wy)

    def _widget_to_norm(self, wx: float, wy: float) -> Tuple[float, float]:
        vr = self._video_rect()
        if vr.width() <= 0 or vr.height() <= 0:
            return (0.0, 0.0)
        nx = (wx - vr.x()) / vr.width()
        ny = (wy - vr.y()) / vr.height()
        return (max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny)))

    def get_roi_pixels(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        return (
            int(self.roi_left * frame_w),
            int(self.roi_top * frame_h),
            int(self.roi_right * frame_w),
            int(self.roi_bottom * frame_h),
        )

    def get_gate_line_pixels(self, frame_w: int, frame_h: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (
            (int(self.gate_p1[0] * frame_w), int(self.gate_p1[1] * frame_h)),
            (int(self.gate_p2[0] * frame_w), int(self.gate_p2[1] * frame_h)),
        )

    # ── Paint Event ──────────────────────────────────────────────────────────
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        vr = self._video_rect()

        # 1. Background video frame
        if self._pixmap and not self._pixmap.isNull():
            painter.drawPixmap(vr.toRect(), self._pixmap)
        else:
            # Placeholder text
            painter.fillRect(self.rect(), QColor("#141414"))
            painter.setPen(QColor("#666666"))
            painter.setFont(QFont("Segoe UI", 14, QFont.Bold))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "Select a video file to view preview and set Tracking Area & Entry Gate line"
            )

        if not self.show_guides_during_play and self._drag_target is None:
            # If guides hidden during play, just show stats badge
            self._draw_stats_overlay(painter, vr)
            return

        # 2. Draw Active Tracking Area (ROI)
        if self.roi_enabled:
            self._draw_roi(painter, vr)

        # 3. Draw Entry Gate Line (Tripwire)
        if self.gate_enabled:
            self._draw_gate_line(painter, vr)

        # 4. Draw HUD / Stats Overlay Banner
        self._draw_stats_overlay(painter, vr)

    def _draw_roi(self, painter: QPainter, vr: QRectF):
        # Convert normalised coords to widget coords
        top_left = self._norm_to_widget(self.roi_left, self.roi_top)
        bot_right = self._norm_to_widget(self.roi_right, self.roi_bottom)

        roi_rect = QRectF(top_left, bot_right)

        # Shaded dim mask outside ROI area
        mask_color = QColor(0, 0, 0, 110)
        # Top mask
        painter.fillRect(QRectF(vr.x(), vr.y(), vr.width(), roi_rect.top() - vr.y()), mask_color)
        # Bottom mask
        painter.fillRect(QRectF(vr.x(), roi_rect.bottom(), vr.width(), vr.bottom() - roi_rect.bottom()), mask_color)
        # Left mask
        painter.fillRect(QRectF(vr.x(), roi_rect.top(), roi_rect.left() - vr.x(), roi_rect.height()), mask_color)
        # Right mask
        painter.fillRect(QRectF(roi_rect.right(), roi_rect.top(), vr.right() - roi_rect.right(), roi_rect.height()), mask_color)

        # Draw ROI Boundary Box
        roi_pen = QPen(QColor(0, 229, 255, 230), 2, Qt.DashLine)
        painter.setPen(roi_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(roi_rect)

        # Draggable boundary handle badges on edges
        font = QFont("Segoe UI", 8, QFont.Bold)
        painter.setFont(font)

        def draw_badge(x, y, text):
            pad = 4
            tw = painter.fontMetrics().horizontalAdvance(text) + pad * 2
            th = painter.fontMetrics().height() + pad
            badge_r = QRectF(x - tw / 2, y - th / 2, tw, th)
            painter.fillRect(badge_r, QColor(10, 15, 25, 220))
            painter.setPen(QColor(0, 229, 255))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(badge_r)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(badge_r, Qt.AlignCenter, text)

        # Handle squares on 4 corners
        h_size = 8
        painter.setBrush(QColor(0, 229, 255))
        painter.setPen(QColor(0, 0, 0))
        for pt in [top_left, QPointF(bot_right.x(), top_left.y()),
                   QPointF(top_left.x(), bot_right.y()), bot_right]:
            painter.drawRect(QRectF(pt.x() - h_size / 2, pt.y() - h_size / 2, h_size, h_size))

        # Midpoint edge handles
        mid_top = QPointF(roi_rect.center().x(), roi_rect.top())
        mid_bot = QPointF(roi_rect.center().x(), roi_rect.bottom())
        mid_left = QPointF(roi_rect.left(), roi_rect.center().y())
        mid_right = QPointF(roi_rect.right(), roi_rect.center().y())

        for pt in [mid_top, mid_bot, mid_left, mid_right]:
            painter.drawRect(QRectF(pt.x() - h_size / 2, pt.y() - h_size / 2, h_size, h_size))

        # Title label at top-left of ROI
        draw_badge(top_left.x() + 75, top_left.y() + 12, "🎯 ACTIVE TRACKING ZONE")

    def _draw_gate_line(self, painter: QPainter, vr: QRectF):
        p1 = self._norm_to_widget(self.gate_p1[0], self.gate_p1[1])
        p2 = self._norm_to_widget(self.gate_p2[0], self.gate_p2[1])

        # Main gate line (Glowing Amber/Yellow)
        line_pen = QPen(QColor(255, 179, 0, 240), 3, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(line_pen)
        painter.drawLine(p1, p2)

        # Direction indicator arrows (perpendicular to line)
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        length = math.hypot(dx, dy)
        if length > 20:
            nx = -dy / length
            ny = dx / length

            # Direction of IN
            in_factor = 1.0 if self.gate_direction == "below" else -1.0
            mid = QPointF((p1.x() + p2.x()) / 2.0, (p1.y() + p2.y()) / 2.0)

            arrow_len = 28.0
            # IN Arrow
            in_tip = QPointF(mid.x() + nx * arrow_len * in_factor,
                             mid.y() + ny * arrow_len * in_factor)
            arrow_pen = QPen(QColor(0, 230, 118), 2)
            painter.setPen(arrow_pen)
            painter.drawLine(mid, in_tip)

            # Arrow head
            arrow_poly = QPolygonF([
                in_tip,
                QPointF(in_tip.x() - (nx * 6 - ny * 5) * in_factor,
                        in_tip.y() - (ny * 6 + nx * 5) * in_factor),
                QPointF(in_tip.x() - (nx * 6 + ny * 5) * in_factor,
                        in_tip.y() - (ny * 6 - nx * 5) * in_factor),
            ])
            painter.setBrush(QColor(0, 230, 118))
            painter.drawPolygon(arrow_poly)

            # Label for IN & OUT
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.setPen(QColor(0, 230, 118))
            painter.drawText(QPointF(in_tip.x() + 6, in_tip.y() + 4), "IN")

            # OUT Arrow (opposite side)
            out_tip = QPointF(mid.x() - nx * arrow_len * in_factor,
                              mid.y() - ny * arrow_len * in_factor)
            painter.setPen(QPen(QColor(255, 82, 82), 2))
            painter.drawLine(mid, out_tip)
            out_poly = QPolygonF([
                out_tip,
                QPointF(out_tip.x() + (nx * 6 + ny * 5) * in_factor,
                        out_tip.y() + (ny * 6 - nx * 5) * in_factor),
                QPointF(out_tip.x() + (nx * 6 - ny * 5) * in_factor,
                        out_tip.y() + (ny * 6 + nx * 5) * in_factor),
            ])
            painter.setBrush(QColor(255, 82, 82))
            painter.drawPolygon(out_poly)
            painter.setPen(QColor(255, 82, 82))
            painter.drawText(QPointF(out_tip.x() + 6, out_tip.y() + 4), "OUT")

        # Draggable Endpoints (Circles with handles)
        handle_rad = 8.0
        painter.setBrush(QColor(255, 179, 0))
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawEllipse(p1, handle_rad, handle_rad)
        painter.drawEllipse(p2, handle_rad, handle_rad)

        # Midpoint handle
        mid_pt = QPointF((p1.x() + p2.x()) / 2.0, (p1.y() + p2.y()) / 2.0)
        painter.drawEllipse(mid_pt, 5.0, 5.0)

        # Badge text centered on line
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        text = "🚪 ENTRY GATE LINE (DRAGGABLE)"
        tw = painter.fontMetrics().horizontalAdvance(text) + 8
        th = painter.fontMetrics().height() + 4
        badge_r = QRectF(mid_pt.x() - tw / 2, mid_pt.y() - th - 12, tw, th)
        painter.fillRect(badge_r, QColor(20, 15, 0, 210))
        painter.setPen(QColor(255, 179, 0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(badge_r)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(badge_r, Qt.AlignCenter, text)

    def _draw_stats_overlay(self, painter: QPainter, vr: QRectF):
        # Live HUD Counter banner at top of canvas
        hud_w = 560
        hud_h = 32
        hud_x = vr.x() + (vr.width() - hud_w) / 2
        hud_y = vr.y() + 8

        badge_r = QRectF(hud_x, hud_y, hud_w, hud_h)
        painter.fillRect(badge_r, QColor(10, 10, 15, 220))
        painter.setPen(QPen(QColor(255, 179, 0), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(badge_r)

        occupancy = max(0, self.live_in_count - self.live_out_count)
        stat_txt = (
            f"🚪 GATE COUNT:  IN: {self.live_in_count}   |   OUT: {self.live_out_count}   "
            f"|   INSIDE: {occupancy}   |   GROUPS: {self.live_group_count}"
        )
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(badge_r, Qt.AlignCenter, stat_txt)

    # ── Mouse Interaction & Dragging ──────────────────────────────────────────
    def _hit_test(self, pt: QPointF) -> Optional[str]:
        tol = 12.0  # Pixel tolerance for grabbing handles/lines

        # Test gate endpoints first (highest priority)
        gp1 = self._norm_to_widget(self.gate_p1[0], self.gate_p1[1])
        gp2 = self._norm_to_widget(self.gate_p2[0], self.gate_p2[1])

        if math.hypot(pt.x() - gp1.x(), pt.y() - gp1.y()) <= tol:
            return "gate_p1"
        if math.hypot(pt.x() - gp2.x(), pt.y() - gp2.y()) <= tol:
            return "gate_p2"

        # Test gate line body
        dist_gate = self._dist_point_to_segment(pt, gp1, gp2)
        if dist_gate <= tol:
            return "gate_line_body"

        # Test ROI edges
        top_left = self._norm_to_widget(self.roi_left, self.roi_top)
        bot_right = self._norm_to_widget(self.roi_right, self.roi_bottom)

        roi_l = top_left.x()
        roi_r = bot_right.x()
        roi_t = top_left.y()
        roi_b = bot_right.y()

        # Corners
        if math.hypot(pt.x() - roi_l, pt.y() - roi_t) <= tol:
            return "roi_top_left"
        if math.hypot(pt.x() - roi_r, pt.y() - roi_t) <= tol:
            return "roi_top_right"
        if math.hypot(pt.x() - roi_l, pt.y() - roi_b) <= tol:
            return "roi_bottom_left"
        if math.hypot(pt.x() - roi_r, pt.y() - roi_b) <= tol:
            return "roi_bottom_right"

        # Edges
        if abs(pt.x() - roi_l) <= tol and (roi_t - tol <= pt.y() <= roi_b + tol):
            return "roi_left"
        if abs(pt.x() - roi_r) <= tol and (roi_t - tol <= pt.y() <= roi_b + tol):
            return "roi_right"
        if abs(pt.y() - roi_t) <= tol and (roi_l - tol <= pt.x() <= roi_r + tol):
            return "roi_top"
        if abs(pt.y() - roi_b) <= tol and (roi_l - tol <= pt.x() <= roi_r + tol):
            return "roi_bottom"

        return None

    @staticmethod
    def _dist_point_to_segment(p: QPointF, a: QPointF, b: QPointF) -> float:
        l2 = (b.x() - a.x()) ** 2 + (b.y() - a.y()) ** 2
        if l2 == 0:
            return math.hypot(p.x() - a.x(), p.y() - a.y())
        t = max(0.0, min(1.0, ((p.x() - a.x()) * (b.x() - a.x()) + (p.y() - a.y()) * (b.y() - a.y())) / l2))
        proj = QPointF(a.x() + t * (b.x() - a.x()), a.y() + t * (b.y() - a.y()))
        return math.hypot(p.x() - proj.x(), p.y() - proj.y())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            target = self._hit_test(event.position())
            if target:
                self._drag_target = target
                self._drag_start_pos = event.position()
                self._drag_initial_values = {
                    "roi_left": self.roi_left,
                    "roi_top": self.roi_top,
                    "roi_right": self.roi_right,
                    "roi_bottom": self.roi_bottom,
                    "gate_p1": self.gate_p1,
                    "gate_p2": self.gate_p2,
                }
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position()
        vr = self._video_rect()

        if self._drag_target and self._drag_start_pos and vr.width() > 0 and vr.height() > 0:
            # Delta in normalised coords
            dnx = (pos.x() - self._drag_start_pos.x()) / vr.width()
            dny = (pos.y() - self._drag_start_pos.y()) / vr.height()

            init = self._drag_initial_values

            if self._drag_target == "gate_p1":
                nx, ny = self._widget_to_norm(pos.x(), pos.y())
                self.gate_p1 = (nx, ny)
                self.gate_line_changed.emit(self.gate_p1[0], self.gate_p1[1], self.gate_p2[0], self.gate_p2[1])

            elif self._drag_target == "gate_p2":
                nx, ny = self._widget_to_norm(pos.x(), pos.y())
                self.gate_p2 = (nx, ny)
                self.gate_line_changed.emit(self.gate_p1[0], self.gate_p1[1], self.gate_p2[0], self.gate_p2[1])

            elif self._drag_target == "gate_line_body":
                ip1 = init["gate_p1"]
                ip2 = init["gate_p2"]
                self.gate_p1 = (max(0.0, min(1.0, ip1[0] + dnx)), max(0.0, min(1.0, ip1[1] + dny)))
                self.gate_p2 = (max(0.0, min(1.0, ip2[0] + dnx)), max(0.0, min(1.0, ip2[1] + dny)))
                self.gate_line_changed.emit(self.gate_p1[0], self.gate_p1[1], self.gate_p2[0], self.gate_p2[1])

            elif self._drag_target == "roi_left":
                new_l = max(0.0, min(self.roi_right - 0.05, init["roi_left"] + dnx))
                self.roi_left = new_l
                self.roi_changed.emit(self.roi_left, self.roi_top, self.roi_right, self.roi_bottom)

            elif self._drag_target == "roi_right":
                new_r = min(1.0, max(self.roi_left + 0.05, init["roi_right"] + dnx))
                self.roi_right = new_r
                self.roi_changed.emit(self.roi_left, self.roi_top, self.roi_right, self.roi_bottom)

            elif self._drag_target == "roi_top":
                new_t = max(0.0, min(self.roi_bottom - 0.05, init["roi_top"] + dny))
                self.roi_top = new_t
                self.roi_changed.emit(self.roi_left, self.roi_top, self.roi_right, self.roi_bottom)

            elif self._drag_target == "roi_bottom":
                new_b = min(1.0, max(self.roi_top + 0.05, init["roi_bottom"] + dny))
                self.roi_bottom = new_b
                self.roi_changed.emit(self.roi_left, self.roi_top, self.roi_right, self.roi_bottom)

            elif self._drag_target == "roi_top_left":
                self.roi_left = max(0.0, min(self.roi_right - 0.05, init["roi_left"] + dnx))
                self.roi_top = max(0.0, min(self.roi_bottom - 0.05, init["roi_top"] + dny))
                self.roi_changed.emit(self.roi_left, self.roi_top, self.roi_right, self.roi_bottom)

            elif self._drag_target == "roi_bottom_right":
                self.roi_right = min(1.0, max(self.roi_left + 0.05, init["roi_right"] + dnx))
                self.roi_bottom = min(1.0, max(self.roi_top + 0.05, init["roi_bottom"] + dny))
                self.roi_changed.emit(self.roi_left, self.roi_top, self.roi_right, self.roi_bottom)

            self.update()
            event.accept()
            return

        # Cursor update based on hover
        hit = self._hit_test(pos)
        if hit in ("gate_p1", "gate_p2"):
            self.setCursor(QCursor(Qt.PointingHandCursor))
        elif hit == "gate_line_body":
            self.setCursor(QCursor(Qt.SizeAllCursor))
        elif hit in ("roi_left", "roi_right"):
            self.setCursor(QCursor(Qt.SizeHorCursor))
        elif hit in ("roi_top", "roi_bottom"):
            self.setCursor(QCursor(Qt.SizeVerCursor))
        elif hit in ("roi_top_left", "roi_bottom_right"):
            self.setCursor(QCursor(Qt.SizeFDiagCursor))
        elif hit in ("roi_top_right", "roi_bottom_left"):
            self.setCursor(QCursor(Qt.SizeBDiagCursor))
        else:
            self.setCursor(QCursor(Qt.ArrowCursor))

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_target = None
            self._drag_start_pos = None
            self.setCursor(QCursor(Qt.ArrowCursor))
            event.accept()
            return
        super().mouseReleaseEvent(event)
