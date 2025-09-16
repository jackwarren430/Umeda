from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QPainter, QPen, QColor, QImage, QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QColorDialog, QFileDialog, QSlider, QLabel, QCheckBox,
    QMessageBox, QLineEdit
)
from ai_image_tool import generate_image_with_gpt5
from ai_objects_tool import generate_game_objects_module 

import tempfile, os, traceback, sys, subprocess  
from pathlib import Path


LAUNCH_BOILER_AFTER_QT: tuple[str, list[str]] | None = None  # (shell_path, args)

class Canvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StaticContents)
        self.setMouseTracking(True)
        self.image = QImage(self.size(), QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)

        self.drawing = False
        self.last_pos = QPoint()
        self.brush_color = QColor(0, 0, 0)
        self.brush_size = 4
        self.eraser = False

    def set_brush_color(self, color: QColor):
        self.brush_color = color

    def set_brush_size(self, size: int):
        self.brush_size = max(1, size)

    def set_eraser(self, on: bool):
        self.eraser = on

    def clear(self):
        self.image.fill(Qt.GlobalColor.white)
        self.update()

    def save_png(self, path: str):
        self.image.save(path, "PNG")

    # load an external image and place it on the canvas
    def load_image(self, path: str):
        src = QImage(path)
        if src.isNull():
            return
        if self.width() <= 0 or self.height() <= 0:
            target_size = (800, 600)
            self.image = QImage(*target_size, QImage.Format.Format_RGB32)
        else:
            self.image = QImage(self.size(), QImage.Format.Format_RGB32)

        self.image.fill(Qt.GlobalColor.white)
        scaled = src.scaled(self.image.size(), Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        x = (self.image.width() - scaled.width()) // 2
        y = (self.image.height() - scaled.height()) // 2

        painter = QPainter(self.image)
        painter.drawImage(x, y, scaled)
        painter.end()
        self.update()

    def resizeEvent(self, event):
        if self.width() > 0 and self.height() > 0:
            new_img = QImage(self.size(), QImage.Format.Format_RGB32)
            new_img.fill(Qt.GlobalColor.white)
            painter = QPainter(new_img)
            painter.drawImage(0, 0, self.image)
            painter.end()
            self.image = new_img
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_pos = event.position().toPoint()
            self.draw_line_to(self.last_pos)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            current_pos = event.position().toPoint()
            self.draw_line_to(current_pos)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False

    def draw_line_to(self, end_pos: QPoint):
        painter = QPainter(self.image)
        color = QColor(Qt.GlobalColor.white) if self.eraser else self.brush_color
        pen = QPen(color, self.brush_size, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(self.last_pos, end_pos)
        painter.end()
        self.last_pos = end_pos
        self.update()

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(0, 0, self.image)
        canvas_painter.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Draw (PyQt6)")
        self.canvas = Canvas()
        self.setCentralWidget(QWidget())
        layout = QVBoxLayout(self.centralWidget())

        # Controls
        controls = QHBoxLayout()

        color_btn = QPushButton("Color")
        color_lbl = QLabel("Brush size:")
        size_slider = QSlider(Qt.Orientation.Horizontal)
        size_slider.setRange(1, 50)
        size_slider.setValue(4)

        eraser_chk = QCheckBox("Eraser")
        clear_btn = QPushButton("Clear")
        save_btn = QPushButton("Save PNG")
        import_btn = QPushButton("Import Image")
        ai_fix_btn = QPushButton("AI Fix")
        ai_playable_btn = QPushButton("AI Playable")

        hint_lbl = QLabel("AI hint:")
        self.ai_hint_input = QLineEdit()
        self.ai_hint_input.setPlaceholderText("e.g., 'This is a piano; map keys to pitches.'")
        self.ai_hint_input.setClearButtonEnabled(True)
        self.ai_hint_input.setMinimumWidth(260)



        controls.addWidget(color_btn)
        controls.addWidget(color_lbl)
        controls.addWidget(size_slider)
        controls.addWidget(eraser_chk)
        controls.addStretch(1)
        controls.addWidget(hint_lbl)                    # NEW
        controls.addWidget(self.ai_hint_input, 1)
        controls.addWidget(import_btn)
        controls.addWidget(ai_fix_btn)
        controls.addWidget(ai_playable_btn)  # NEW
        controls.addWidget(clear_btn)
        controls.addWidget(save_btn)

        layout.addLayout(controls)
        layout.addWidget(self.canvas, stretch=1)

        # Wire up
        color_btn.clicked.connect(self.pick_color)
        size_slider.valueChanged.connect(self.canvas.set_brush_size)
        eraser_chk.toggled.connect(self.canvas.set_eraser)
        clear_btn.clicked.connect(self.canvas.clear)
        save_btn.clicked.connect(self.save_png)
        import_btn.clicked.connect(self.import_image)
        ai_fix_btn.clicked.connect(self.ai_fix_image)
        ai_playable_btn.clicked.connect(self.ai_playable_mode)


        # Menu/shortcuts
        save_action = QAction("Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_png)
        self.addAction(save_action)

        clear_action = QAction("Clear", self)
        clear_action.setShortcut(QKeySequence("Ctrl+N"))
        clear_action.triggered.connect(self.canvas.clear)
        self.addAction(clear_action)

        open_action = QAction("Open", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.import_image)
        self.addAction(open_action)

        ai_action = QAction("AI Fix", self)
        ai_action.setShortcut(QKeySequence("Ctrl+Shift+F"))
        ai_action.triggered.connect(self.ai_fix_image)
        self.addAction(ai_action)

        ai_playable_action = QAction("AI Playable", self)
        ai_playable_action.setShortcuts([QKeySequence("Ctrl+Shift+G"), QKeySequence("Meta+Shift+G")])
        ai_playable_action.triggered.connect(self.ai_playable_mode); self.addAction(ai_playable_action)




        self.resize(900, 600)

    def pick_color(self):
        color = QColorDialog.getColor(self.canvas.brush_color, self, "Pick Color")
        if color.isValid():
            self.canvas.set_brush_color(color)

    def save_png(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "drawing.png",
            "PNG Images (*.png)")
        if path:
            if not path.lower().endswith(".png"):
                path += ".png"
            self.canvas.save_png(path)

    def import_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Image", "",
            "Images (*.png *.jpg *.jpeg)")
        if path:
            self.canvas.load_image(path)

    def ai_fix_image(self):
        try:
            fd, temp_in_path = tempfile.mkstemp(prefix="canvas_", suffix=".png")
            os.close(fd)
            self.canvas.image.save(temp_in_path, "PNG")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to snapshot canvas:\n{e}")
            return

        prompt = ("Generate a cleaned up version of this image. "
                  "Keep everything the same, just make the lines straight "
                  "(or smoothly curved when appropriate) and the text look nice.")
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            result = generate_image_with_gpt5(temp_in_path, prompt)
            png_path = result[0] if isinstance(result, tuple) else result
            if not png_path or not os.path.exists(png_path):
                raise RuntimeError("AI did not return a valid PNG path.")
            self.canvas.load_image(png_path)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "AI Fix Failed", f"{e}")
        finally:
            QApplication.restoreOverrideCursor()
            try:
                os.remove(temp_in_path)
            except Exception:
                pass

    def ai_playable_mode(self):
        # 1) Snapshot canvas to a temp PNG
        try:
            fd, temp_png = tempfile.mkstemp(prefix="canvas_boiler_", suffix=".png")
            os.close(fd)
            self.canvas.image.save(temp_png, "PNG")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to snapshot canvas:\n{e}")
            return

        user_hint = self.ai_hint_input.text().strip()

        # 2) Ask GPT-5 to generate a *module* that implements create_game(...)
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            module_path = generate_game_objects_module(
                temp_png,
                user_hint,
                out_dir="games",
                base_name="objects"
            )
        except Exception as e:
            traceback.print_exc()
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "AI Play Failed", f"{e}")
            try: os.remove(temp_png)
            except Exception: pass
            return
        finally:
            QApplication.restoreOverrideCursor()

        # 3) Launch boilerplate runner with the PNG + generated module
        shell_path = str(Path(__file__).resolve().parent / "game_shell.py")
        if not os.path.isfile(shell_path):
            QMessageBox.critical(self, "Error", f"game_shell.py not found at:\n{shell_path}")
            try: os.remove(temp_png)
            except Exception: pass
            return

        args = [temp_png, "--module", module_path]
        global LAUNCH_BOILER_AFTER_QT
        LAUNCH_BOILER_AFTER_QT = (shell_path, args)

        # Close Qt and hand off to the boiler
        self.close()
        QApplication.instance().quit()



def main():
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    exit_code = app.exec()

    global LAUNCH_BOILER_AFTER_QT
    if LAUNCH_BOILER_AFTER_QT:
        shell_path, args = LAUNCH_BOILER_AFTER_QT
        try:
            subprocess.run([sys.executable, shell_path, *args], check=False)
        finally:
            try: os.remove(args[0])  # remove the temp PNG snapshot
            except Exception: pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()