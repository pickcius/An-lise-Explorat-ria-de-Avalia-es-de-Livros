import re
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
import plotly.express as px
from django.conf import settings
from django.shortcuts import render
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

DATASET_PATH = Path(settings.BASE_DIR) / 'books-15k.csv'

try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception:
    pass


def parse_helpfulness(value):
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if not text or text == 'nan':
        return 0.0
    if '/' not in text:
        try:
            return float(text)
        except ValueError:
            return 0.0
    try:
        helpful, total = text.split('/', 1)
        helpful = float(helpful)
        total = float(total)
        if total == 0:
            return 0.0
        return helpful / total
    except ValueError:
        return 0.0


def generate_wordcloud(df):
    text_values = df['review/text'].fillna('').astype(str).str.lower()
    text = ' '.join(text_values)
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = [word for word in text.split() if len(word) > 2 and word not in {'book', 'books', 'read', 'reads', 'reading', 'story', 'people', 'really', 'would'}]
    if not words:
        words = ['book', 'reading', 'review']
    wc = WordCloud(width=900, height=400, background_color='white', colormap='viridis')
    wc.generate(' '.join(words))
    output_path = Path(settings.BASE_DIR) / 'core' / 'static' / 'core' / 'wordcloud.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wc.to_file(str(output_path))
    return output_path.name


def build_dashboard_context():
    df = pd.read_csv(DATASET_PATH)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['review/score'] = pd.to_numeric(df['review/score'], errors='coerce')
    df['review/time'] = pd.to_numeric(df['review/time'], errors='coerce')
    df['review/date'] = pd.to_datetime(df['review/time'], unit='s', errors='coerce')
    df['review/helpfulness_ratio'] = df['review/helpfulness'].apply(parse_helpfulness)

    missing = df.isna().sum().sort_values(ascending=False)
    missing_table = missing.reset_index()
    missing_table.columns = ['Coluna', 'Quantidade']
    missing_table['Proporcao'] = (missing_table['Quantidade'] / len(df) * 100).round(2)

    price_mean = round(float(df['Price'].mean()), 2)
    books_zero_price = int((df['Price'] == 0).sum())
    score_mean = float(df['review/score'].mean())
    score_median = float(df['review/score'].median())
    score_std = float(df['review/score'].std())
    total_unique_books = df['Title'].nunique()
    mean_reviews_per_book = float(df['Title'].value_counts().mean())

    score_counts = df['review/score'].value_counts().sort_index().rename_axis('Nota').reset_index(name='Frequencia')
    score_counts['Nota'] = score_counts['Nota'].astype(int)

    top_books = df['Title'].value_counts().head(10).reset_index()
    top_books.columns = ['Titulo', 'Avaliacoes']

    top_users = df['User_id'].value_counts().head(10).reset_index()
    top_users.columns = ['Usuario', 'Avaliacoes']

    useful_reviews = df.sort_values('review/helpfulness_ratio', ascending=False).head(5)[
        ['Title', 'profileName', 'review/helpfulness', 'review/helpfulness_ratio', 'review/score', 'review/summary']
    ].copy()
    useful_reviews.columns = ['Titulo', 'Usuario', 'Utilidade', 'TaxaUtilidade', 'Nota', 'Resumo']
    useful_reviews['TaxaUtilidade'] = useful_reviews['TaxaUtilidade'].round(2)

    years = df['review/date'].dt.year.value_counts().sort_index()
    year_df = years.reset_index()
    year_df.columns = ['Ano', 'Avaliacoes']

    score_chart = px.bar(
        score_counts,
        x='Nota',
        y='Frequencia',
        title='Distribuição das notas (1 a 5)',
        labels={'Nota': 'Nota', 'Frequencia': 'Quantidade'},
        color='Nota',
        color_continuous_scale='Viridis'
    )
    score_chart.update_layout(template='plotly_white', height=420, margin=dict(l=20, r=20, t=50, b=10))

    price_score_chart = px.scatter(
        df,
        x='Price',
        y='review/score',
        opacity=0.5,
        title='Preço vs. nota atribuída',
        labels={'Price': 'Preço do livro', 'review/score': 'Nota'},
        color='review/score',
        color_continuous_scale='Viridis'
    )
    price_score_chart.update_layout(template='plotly_white', height=420, margin=dict(l=20, r=20, t=50, b=10))

    time_chart = px.line(
        year_df,
        x='Ano',
        y='Avaliacoes',
        title='Evolução temporal das avaliações por ano',
        markers=True,
        line_shape='spline'
    )
    time_chart.update_layout(template='plotly_white', height=420, margin=dict(l=20, r=20, t=50, b=10))

    type_counts = df.dtypes.astype(str).value_counts().reset_index()
    type_counts.columns = ['Tipo', 'Quantidade']
    structure_chart = px.bar(
        type_counts,
        x='Tipo',
        y='Quantidade',
        title='Tipos de dados no dataset',
        labels={'Tipo': 'Tipo de dado', 'Quantidade': 'Quantidade de colunas'},
        color='Tipo',
        color_discrete_sequence=['#d9b250', '#ef4444', '#f59e0b', '#fca5a5', '#b91c1c']
    )
    structure_chart.update_layout(template='plotly_white', height=360, margin=dict(l=20, r=20, t=50, b=10))

    sentiment = SentimentIntensityAnalyzer()
    sentiment_labels = []
    for text in df['review/text'].fillna('').astype(str).head(5000):
        score = sentiment.polarity_scores(text)
        if score['compound'] > 0.05:
            sentiment_labels.append('Positivo')
        elif score['compound'] < -0.05:
            sentiment_labels.append('Negativo')
        else:
            sentiment_labels.append('Neutro')
    sentiment_counts = Counter(sentiment_labels)
    sentiment_counts = {
        'Positivo': sentiment_counts.get('Positivo', 0),
        'Neutro': sentiment_counts.get('Neutro', 0),
        'Negativo': sentiment_counts.get('Negativo', 0),
    }

    generate_wordcloud(df)

    context = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'dtypes': df.dtypes.astype(str).reset_index().rename(columns={'index': 'Coluna', 0: 'Tipo'}),
        'missing_table': missing_table.to_dict(orient='records'),
        'price_mean': price_mean,
        'books_zero_price': books_zero_price,
        'score_mean': round(score_mean, 2),
        'score_median': round(score_median, 2),
        'score_std': round(score_std, 2),
        'total_unique_books': total_unique_books,
        'mean_reviews_per_book': round(mean_reviews_per_book, 2),
        'top_books': top_books.to_dict(orient='records'),
        'top_users': top_users.to_dict(orient='records'),
        'useful_reviews': useful_reviews.to_dict(orient='records'),
        'score_counts': score_counts.to_dict(orient='records'),
        'year_data': year_df.to_dict(orient='records'),
        'score_chart': score_chart.to_html(full_html=False, include_plotlyjs='cdn'),
        'price_score_chart': price_score_chart.to_html(full_html=False, include_plotlyjs='cdn'),
        'time_chart': time_chart.to_html(full_html=False, include_plotlyjs='cdn'),
        'structure_chart': structure_chart.to_html(full_html=False, include_plotlyjs='cdn'),
        'sentiment_counts': sentiment_counts,
        'positive_reviews_ratio': round((sentiment_counts['Positivo'] / sum(sentiment_counts.values())) * 100, 2),
        'negative_reviews_ratio': round((sentiment_counts['Negativo'] / sum(sentiment_counts.values())) * 100, 2),
    }
    return context


def compute_sentiment_summary(df, sample_size=5000):
    analyzer = SentimentIntensityAnalyzer()
    labels = []
    for text in df['review/text'].fillna('').astype(str).head(sample_size):
        score = analyzer.polarity_scores(text)
        if score['compound'] > 0.05:
            labels.append('Positivo')
        elif score['compound'] < -0.05:
            labels.append('Negativo')
        else:
            labels.append('Neutro')

    count = Counter(labels)
    total = sum(count.values())
    return {
        'Positivo': count.get('Positivo', 0),
        'Neutro': count.get('Neutro', 0),
        'Negativo': count.get('Negativo', 0),
        'total': total,
        'percentual_positivo': round((count.get('Positivo', 0) / total) * 100, 2) if total else 0,
        'percentual_negativo': round((count.get('Negativo', 0) / total) * 100, 2) if total else 0,
    }


def hub(request):
    return render(request, 'core/hub.html')


def dashboard(request):
    return render(request, 'core/index.html', build_dashboard_context())


def home(request):
    return hub(request)


def sentimento(request):
    df = pd.read_csv(DATASET_PATH)
    context = compute_sentiment_summary(df)
    context['dataset_total'] = len(df)
    context['title'] = 'Análise de Sentimento'
    return render(request, 'core/sentimento.html', context)
