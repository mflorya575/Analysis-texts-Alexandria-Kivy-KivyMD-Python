import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QTextEdit, QLabel, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem
from nltk import WordNetLemmatizer
from textblob import TextBlob
import nltk
from nltk.corpus import wordnet
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd


# Указываем новый путь для данных NLTK
nltk.data.path.append('C:/python/9_analys_texts/data/nltk_data')

# Скачиваем нужные пакеты
nltk.download('wordnet', download_dir='C:/python/9_analys_texts/data/nltk_data')
nltk.download('omw-1.4')  # Чтобы WordNet мог работать с расширенным набором слов

lemmatizer = WordNetLemmatizer()


class TextAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Alexandria")
        self.setGeometry(450, 200, 1000, 600)

        # Create layout and widgets
        self.layout = QVBoxLayout()
        self.label = QLabel("Выберите файл/файлы для анализа:")
        self.button = QPushButton("Открыть файл/файлы")
        self.sentiment_button = QPushButton("Тональность текста")
        self.synonym_button = QPushButton("Синонимы")
        self.lexeme_button = QPushButton("Лексемы")
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)

        self.factor_button = QPushButton("Факторный анализ")
        self.layout.addWidget(self.factor_button)
        self.factor_button.clicked.connect(self.perform_factor_analysis_from_text)

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

        self.texts = []  # Список для хранения текста из нескольких файлов

    def open_file(self):
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(self, "Open Text Files", "", "Text Files (*.txt)")

        if file_paths:
            self.texts = []  # Сброс списка текстов
            for file_path in file_paths:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    self.texts.append(text)  # Добавляем текст в список
                    self.result_display.append(f"Loaded text from: {file_path}\n{text}\n")  # Отображаем загруженный текст

    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            return "Позитивный"
        elif sentiment < 0:
            return "Негативный"
        else:
            return "Нейтральный"

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

    def analyze_sentiment_from_text(self):
        results = []
        for text in self.texts:
            sentiment = self.analyze_sentiment(text)
            results.append(f"Sentiment: {sentiment}")
        self.result_display.setText("\n".join(results))

    def count_synonyms_from_text(self):
        synonym_list = self.get_synonyms('happy')
        synonym_list.update({'cheerful', 'joyful', 'blissful'})
        results = []

        for text in self.texts:
            synonym_counts = self.count_synonyms(text, synonym_list)
            result = f"Synonym counts in text:\n" + "\n".join([f"{synonym} ({count})" for synonym, count in synonym_counts.items() if count > 0])
            results.append(result)

        self.result_display.setText("\n\n".join(results))

    def count_lexemes_from_text(self):
        results = []
        for text in self.texts:
            lexeme_counts = self.count_lexemes(text)
            result = f"Lexeme counts in text:\n" + "\n".join([f"{lexeme} ({count})" for lexeme, count in lexeme_counts])
            results.append(result)

        self.result_display.setText("\n\n".join(results))

    def perform_factor_analysis(self, texts):
        if len(texts) < 2:
            raise ValueError("Недостаточно данных для факторного анализа (необходимо хотя бы 2 документа)")

        # Преобразование текста в числовые данные через TF-IDF
        vectorizer = TfidfVectorizer(max_features=100)

        try:
            X = vectorizer.fit_transform(texts).toarray()  # Преобразуем сразу все тексты
        except Exception as e:
            raise ValueError(f"Ошибка при векторизации текстов: {str(e)}")

        n_samples, n_features = X.shape
        n_components = min(n_samples, n_features)  # Количество факторов не должно превышать min(n_samples, n_features)

        if n_components < 2:
            raise ValueError("Недостаточно данных для факторного анализа после векторизации")

        # Применение метода главных компонент (PCA)
        try:
            pca = PCA(n_components=n_components)  # Устанавливаем допустимое количество факторов
            X_pca = pca.fit_transform(X)
        except Exception as e:
            raise ValueError(f"Ошибка при применении PCA: {str(e)}")

        # Получение признаков (слов)
        feature_names = vectorizer.get_feature_names_out()

        # Создание DataFrame для отображения результатов
        df = pd.DataFrame(X_pca, columns=[f"Factor {i + 1}" for i in range(n_components)])

        return df, feature_names  # Возвращаем и DataFrame, и список признаков

    # Обработка для кнопки факторного анализа
    def perform_factor_analysis_from_text(self):
        try:
            if len(self.texts) < 2:
                raise ValueError("Для факторного анализа необходимо как минимум 2 текста.")

            # Получаем DataFrame с факторами и список признаков
            df_factors, feature_names = self.perform_factor_analysis(self.texts)

            # Отображаем результаты
            self.display_factor_analysis_results(df_factors, feature_names)

        except Exception as e:
            self.result_display.setText(f"Ошибка при выполнении факторного анализа: {str(e)}")

    # Отображение результатов в таблице
    def display_factor_analysis_results(self, df, feature_names):
        # Создаем QTableWidget для отображения результатов факторного анализа
        num_rows = df.shape[0]  # Количество строк = количество документов
        num_columns = df.shape[1] + 1  # +1 для первой колонки с признаками (словами)

        table = QTableWidget()
        table.setRowCount(num_rows)  # Устанавливаем количество строк (количество документов)
        table.setColumnCount(num_columns)  # Устанавливаем количество колонок (количество факторов + 1 для слов)

        # Устанавливаем заголовки столбцов для факторов
        table.setHorizontalHeaderLabels(["Фактор. нагрузки"] + [f"Factor {i + 1}" for i in range(df.shape[1])])

        # Заполняем первую колонку (слева) словами (признаками)
        for i, word in enumerate(feature_names):
            item = QTableWidgetItem(word)
            table.setItem(i, 0, item)  # Признаки будут в первой колонке (индекс 0)

        # Заполняем таблицу значениями факторного анализа
        for i in range(df.shape[0]):  # Для каждой строки документа
            for j in range(df.shape[1]):  # Для каждого фактора
                item = QTableWidgetItem(f"{df.iloc[i, j]:.6f}")
                table.setItem(i, j + 1, item)  # Числа начинаются со второго столбца

        # Устанавливаем созданную таблицу как центральный виджет
        self.setCentralWidget(table)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TextAnalyzerApp()
    window.show()
    sys.exit(app.exec_())
