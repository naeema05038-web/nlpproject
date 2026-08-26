from src.preprocessing import (
    load_data,
    clean_dataset,
    prepare_text,
    split_data,
    create_tfidf
)

df = load_data("data/mental_health_cleaned.csv")

df = clean_dataset(df)
df = prepare_text(df)

X_train, X_test, y_train, y_test = split_data(df)

X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf(
    X_train,
    X_test
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)