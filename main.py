import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QTextEdit, QLabel, QVBoxLayout, QWidget
from nltk import WordNetLemmatizer
from textblob import TextBlob
import nltk
from nltk.corpus import wordnet


# Указываем новый путь для данных NLTK
nltk.data.path.append('C:/python/9_analys_texts/data/nltk_data')

# Скачиваем нужные пакеты
nltk.download('wordnet', download_dir='C:/python/9_analys_texts/data/nltk_data')
nltk.download('omw-1.4')  # Чтобы WordNet мог работать с расширенным набором слов

lemmatizer = WordNetLemmatizer()


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
                # Добавляем как исходные формы слов, так и леммы
                synonyms.add(lemma.name())
        return synonyms

    def count_synonyms(self, text, synonym_list):
        # Приводим текст к нижнему регистру и разбиваем на слова
        word_list = text.lower().split()

        # Лемматизируем каждое слово
        lemmatized_words = [lemmatizer.lemmatize(word) for word in word_list]

        # Подсчитываем количество синонимов
        count = sum(1 for word in lemmatized_words if word in synonym_list)
        return count

    def analyze_text(self, text):
        # Анализ тональности
        sentiment = self.analyze_sentiment(text)

        # Пример анализа синонимов для слова "happy"
        synonym_list = self.get_synonyms('happy')
        synonym_count = self.count_synonyms(text, synonym_list)

        # Отображение результатов
        analysis_result = f"Sentiment: {sentiment}\n"
        analysis_result += f"Synonym count for 'happy' and its synonyms: {synonym_count}"
        self.result_display.setText(analysis_result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TextAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
