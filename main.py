import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QTextEdit, QLabel, QVBoxLayout, QWidget
from nltk import WordNetLemmatizer
from textblob import TextBlob
import nltk
from nltk.corpus import wordnet
import re


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
        # Получаем синонимы из WordNet
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                # Добавляем все леммы (варианты слов)
                synonyms.add(lemma.name().lower())
        return synonyms

    def count_synonyms(self, text, synonym_list):
        # Приводим текст к нижнему регистру и удаляем пунктуацию
        cleaned_text = re.sub(r"[^\w\s]", "", text.lower())
        word_list = cleaned_text.split()

        # Лемматизируем каждое слово для нормализации
        lemmatized_words = [lemmatizer.lemmatize(word) for word in word_list]

        # Подсчитываем количество вхождений каждого синонима
        synonym_count = {synonym: 0 for synonym in synonym_list}
        for word in lemmatized_words:
            if word in synonym_count:
                synonym_count[word] += 1

        return synonym_count

    def count_lexemes(self, text):
        # Удаляем пунктуацию и приводим текст к нижнему регистру
        cleaned_text = re.sub(r"[^\w\s]", "", text.lower())

        # Разбиваем текст на слова
        word_list = cleaned_text.split()

        # Лемматизируем каждое слово для нормализации
        lemmatized_words = [lemmatizer.lemmatize(word) for word in word_list]

        # Подсчитываем количество вхождений каждой лексемы
        lexeme_count = {}
        for word in lemmatized_words:
            if word in lexeme_count:
                lexeme_count[word] += 1
            else:
                lexeme_count[word] = 1

        # Сортируем лексемы по убыванию их частоты
        sorted_lexemes = sorted(lexeme_count.items(), key=lambda item: item[1], reverse=True)

        return sorted_lexemes

    def analyze_text(self, text):
        # Анализ тональности
        sentiment = self.analyze_sentiment(text)

        # Получаем список синонимов для слова "happy"
        synonym_list = self.get_synonyms('happy')
        synonym_list.update({'cheerful', 'joyful', 'blissful'})

        # Считаем синонимы в тексте
        synonym_counts = self.count_synonyms(text, synonym_list)

        # Считаем уникальные лексемы и их частоту
        lexeme_counts = self.count_lexemes(text)

        # Формируем результат анализа
        analysis_result = f"Sentiment: {sentiment}\n"
        analysis_result += "Synonym counts:\n"

        # Добавляем синонимы и их количество в результат
        for synonym, count in synonym_counts.items():
            if count > 0:  # Показать только те синонимы, которые есть в тексте
                analysis_result += f"{synonym} ({count})\n"

        analysis_result += "\nLexeme counts:\n"
        for lexeme, count in lexeme_counts:
            analysis_result += f"{lexeme} ({count})\n"

        self.result_display.setText(analysis_result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TextAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
