import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QTextEdit, QLabel, QVBoxLayout, QWidget
from textblob import TextBlob
import nltk
from nltk.corpus import wordnet


# Указываем новый путь для данных NLTK
nltk.data.path.append('C:/python/9_analys_texts/data/nltk_data')

# Скачиваем нужные пакеты
nltk.download('wordnet', download_dir='C:/python/9_analys_texts/data/nltk_data')


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
        # Анализ тональности текста
        sentiment = self.analyze_sentiment(text)

        # Подсчет синонимов (для примера используем слово 'happy')
        synonym_list = self.get_synonyms('happy')
        synonym_count = self.count_synonyms(text, synonym_list)

        # Формируем результат анализа
        analysis_result = f"Sentiment: {sentiment}\n"
        analysis_result += f"Synonym count for 'happy': {synonym_count}"

        # Отображаем результат анализа
        self.result_display.setText(analysis_result)

    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            return "Positive"
        elif sentiment < 0:
            return "Negative"
        else:
            return "Neutral"

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return synonyms

    def count_synonyms(self, text, synonym_list):
        word_list = text.split()
        count = sum(1 for word in word_list if word in synonym_list)
        return count


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TextAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
