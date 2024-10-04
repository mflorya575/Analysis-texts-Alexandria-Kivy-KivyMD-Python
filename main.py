import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QTextEdit, QLabel, QVBoxLayout, QWidget


class TextAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Text Analyzer")
        self.setGeometry(100, 100, 600, 400)

        # Create layout and widgets
        self.layout = QVBoxLayout()
        self.label = QLabel("Choose a text file to analyze:")
        self.button = QPushButton("Open File")
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)

        # Add widgets to layout
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.result_display)

        # Set central widget
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Connect button to file dialog
        self.button.clicked.connect(self.open_file)

    def open_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Text File", "", "Text Files (*.txt)")

        if file_path:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                self.analyze_text(text)

    def analyze_text(self, text):
        # Placeholder for text analysis logic
        self.result_display.setText("Analysis results will be displayed here.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TextAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
