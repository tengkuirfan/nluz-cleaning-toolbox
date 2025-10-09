import pandas as pd
from nluztoolbox import TextCleaning, TabularCleaning

# Create a sample dataset with mixed data types
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'price': [100, 200, 150, None, 300],
    'quantity': [10, 20, 15, 25, 30],
    'product_name': [
        "Apple iPhone 14 Pro Max",
        "Samsung Galaxy S23 Ultra",
        "Google Pixel 7 Pro",
        "OnePlus 11 5G",
        "Xiaomi Mi 13 Pro"
    ],
    'review': [
        "I CAN'T believe how AMAZING this is! 😍 Check out https://review.com",
        "Don't buy this!!! Visit <b>competitor.com</b> instead.",
        "It's okay, nothing special... Email me at test@example.com",
        "BEST PHONE EVER!!! <p>Worth every penny</p> 💯",
        "I wouldn't recommend it tbh. See more at http://reviews.com"
    ]
})

print("=" * 80)
print("ORIGINAL DATAFRAME")
print("=" * 80)
print(df)
print("\n")

# ============================================================================
# Method 1: Using TextCleaning directly on a DataFrame column
# ============================================================================
print("=" * 80)
print("METHOD 1: Direct TextCleaning on DataFrame Column")
print("=" * 80)

# Clean the review column
df_cleaned = (TextCleaning(df, text_column='review')
              .lowercase()
              .expand_contractions()
              .remove_urls()
              .remove_emails()
              .remove_html_tags()
              .remove_punctuation()
              .remove_whitespace(mode="extra")
              .get())

print(df_cleaned[['id', 'product_name', 'review']])
print("\n")

# ============================================================================
# Method 2: Using TabularCleaning with process_column for text cleaning
# ============================================================================
print("=" * 80)
print("METHOD 2: TabularCleaning.process_column with Text Cleaning Function")
print("=" * 80)

# Define a text cleaning function
def clean_text(text):
    """Custom text cleaning function"""
    return (TextCleaning(str(text))
            .lowercase()
            .expand_contractions()
            .remove_urls()
            .remove_emails()
            .remove_html_tags()
            .remove_punctuation()
            .remove_whitespace(mode="extra")
            .get())

# Use TabularCleaning to apply text cleaning + other operations
df_final = (TabularCleaning(df)
            .handle_missing(columns=['price'], method='mean')  # Handle missing values
            .process_column(column='review', func=clean_text)  # Clean review text
            .process_column(column='product_name', func=clean_text)  # Clean product names
            .get())

print(df_final[['id', 'price', 'product_name', 'review']])
print("\n")

# ============================================================================
# Method 3: Advanced - Separate cleaning for different columns
# ============================================================================
print("=" * 80)
print("METHOD 3: Different Cleaning for Different Columns")
print("=" * 80)

# Light cleaning for product names (keep some structure)
def light_clean(text):
    return (TextCleaning(str(text))
            .remove_urls()
            .remove_html_tags()
            .remove_whitespace(mode="extra")
            .get())

# Heavy cleaning for reviews (prepare for NLP)
def heavy_clean(text):
    try:
        return (TextCleaning(str(text))
                .lowercase()
                .expand_contractions()
                .remove_urls()
                .remove_emails()
                .remove_html_tags()
                .remove_punctuation()
                .remove_whitespace(mode="extra")
                .remove_stopwords(language="english")
                .stem(method="porter")
                .get())
    except:
        # Fallback if NLTK features not available
        return (TextCleaning(str(text))
                .lowercase()
                .expand_contractions()
                .remove_urls()
                .remove_emails()
                .remove_html_tags()
                .remove_punctuation()
                .remove_whitespace(mode="extra")
                .get())

df_advanced = (TabularCleaning(df)
               .handle_missing(columns=['price'], method='median')
               .process_column(column='product_name', func=light_clean)
               .process_column(column='review', func=heavy_clean)
               .get())

print(df_advanced[['id', 'price', 'product_name', 'review']])
print("\n")

# ============================================================================
# Method 4: Create new columns with cleaned text
# ============================================================================
print("=" * 80)
print("METHOD 4: Create New Columns with Cleaned Text")
print("=" * 80)

# Keep original and add cleaned versions
df_with_original = df.copy()

# Clean review and create new column
df_with_original = (TextCleaning(df_with_original, text_column='review')
                    .lowercase()
                    .expand_contractions()
                    .remove_urls()
                    .remove_emails()
                    .remove_html_tags()
                    .remove_punctuation()
                    .remove_whitespace(mode="extra")
                    .get())

# Rename the review column to review_cleaned
df_with_original['review_cleaned'] = df_with_original['review']
df_with_original['review'] = df['review']  # Restore original

print(df_with_original[['id', 'review', 'review_cleaned']])
print("\n")

# ============================================================================
# Method 5: Tokenization on DataFrame
# ============================================================================
print("=" * 80)
print("METHOD 5: Tokenization on DataFrame")
print("=" * 80)

df_tokens = (TextCleaning(df, text_column='review')
             .lowercase()
             .expand_contractions()
             .remove_urls()
             .remove_emails()
             .remove_html_tags()
             .remove_punctuation()
             .tokenize(method="simple")  # Creates 'review_tokens' column
             .get())

print("First 3 rows with tokens:")
for idx, row in df_tokens.head(3).iterrows():
    print(f"\nReview {row['id']}: {df['review'][idx][:50]}...")
    print(f"Tokens: {row['review_tokens']}")

print("\n")

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("SUMMARY: Text Cleaning with DataFrame Columns")
print("=" * 80)
print("""
✓ Method 1: Direct TextCleaning on DataFrame column
  - Simple and straightforward
  - Modifies the specified column directly
  
✓ Method 2: Use with TabularCleaning.process_column
  - Combine text cleaning with other tabular operations
  - Define reusable cleaning functions
  
✓ Method 3: Different cleaning for different columns
  - Apply different cleaning strategies
  - Light vs heavy cleaning
  - With or without NLTK features
  
✓ Method 4: Create new columns
  - Keep original data
  - Add cleaned versions as new columns
  
✓ Method 5: Tokenization on DataFrame
  - Creates new column with tokens
  - Works seamlessly with pandas DataFrames

All methods support:
- Method chaining
- Operation logging
- DataFrame preservation
- Error handling
""")

print("=" * 80)
print("Example complete!")
print("=" * 80)
