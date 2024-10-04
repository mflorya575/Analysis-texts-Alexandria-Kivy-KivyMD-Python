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
        self.setGeometry(450, 200, 1000, 600)

        # Create layout and widgets
        self.layout = QVBoxLayout()
        self.label = QLabel("Choose a text file to analyze:")
        self.button = QPushButton("Open File")
        self.sentiment_button = QPushButton("Analyze Sentiment")
        self.synonym_button = QPushButton("Count Synonyms")
        self.lexeme_button = QPushButton("Count Lexemes")
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)

        # Add widgets to layout
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.sentiment_button)
        self.layout.addWidget(self.synonym_button)
        self.layout.addWidget(self.lexeme_button)
        self.layout.addWidget(self.result_display)

        # Set central widget
        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Connect buttons to functions
        self.button.clicked.connect(self.open_file)
        self.sentiment_button.clicked.connect(self.analyze_sentiment_from_text)
        self.synonym_button.clicked.connect(self.count_synonyms_from_text)
        self.lexeme_button.clicked.connect(self.count_lexemes_from_text)

        self.text = ""  # Сохраняем текст для анализа

    def open_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Text File", "", "Text Files (*.txt)")

        if file_path:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.text = file.read()
                self.result_display.setText(self.text)  # Отображаем загруженный текст

    def analyze_sentiment(self):
        blob = TextBlob(self.text)
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
                synonyms.add(lemma.name().lower())
        return synonyms

    def count_synonyms(self, text, synonym_list):
        cleaned_text = re.sub(r"[^\w\s]", "", text.lower())
        word_list = cleaned_text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in word_list]

        synonym_count = {synonym: 0 for synonym in synonym_list}
        for word in lemmatized_words:
            if word in synonym_count:
                synonym_count[word] += 1

        return synonym_count

    def count_lexemes(self, text):
        cleaned_text = re.sub(r"[^\w\s]", "", text.lower())
        word_list = cleaned_text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in word_list]

        lexeme_count = {}
        for word in lemmatized_words:
            if word in lexeme_count:
                lexeme_count[word] += 1
            else:
                lexeme_count[word] = 1

        sorted_lexemes = sorted(lexeme_count.items(), key=lambda item: item[1], reverse=True)
        return sorted_lexemes

    def analyze_text(self):
        sentiment = self.analyze_sentiment()
        synonym_list = self.get_synonyms('happy')
        synonym_list.update({'cheerful', 'joyful', 'blissful'})
        synonym_counts = self.count_synonyms(self.text, synonym_list)
        lexeme_counts = self.count_lexemes(self.text)

        analysis_result = f"Sentiment: {sentiment}\n"
        analysis_result += "Synonym counts:\n"
        for synonym, count in synonym_counts.items():
            if count > 0:
                analysis_result += f"{synonym} ({count})\n"

        analysis_result += "\nLexeme counts:\n"
        for lexeme, count in lexeme_counts:
            analysis_result += f"{lexeme} ({count})\n"

        self.result_display.setText(analysis_result)

    def analyze_sentiment_from_text(self):
        sentiment = self.analyze_sentiment()
        self.result_display.setText(f"Sentiment: {sentiment}")

    def count_synonyms_from_text(self):
        synonym_list = self.get_synonyms('happy')
        synonym_list.update({'cheerful', 'joyful', 'blissful'})
        synonym_counts = self.count_synonyms(self.text, synonym_list)
        result = "\n".join([f"{synonym} ({count})" for synonym, count in synonym_counts.items() if count > 0])
        self.result_display.setText(f"Synonym counts:\n{result}")

    def count_lexemes_from_text(self):
        lexeme_counts = self.count_lexemes(self.text)
        result = "\n".join([f"{lexeme} ({count})" for lexeme, count in lexeme_counts])
        self.result_display.setText(f"Lexeme counts:\n{result}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TextAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
