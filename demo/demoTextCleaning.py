import pandas as pd
from nluztoolbox import TextCleaning, get_contractions_dict

print("TEXT CLEANING DEMO")
print("\nPART 1: SINGLE TEXT STRING EXAMPLES")

sample_text = "  Hello WORLD! I can't believe it's 2024. Visit https://example.com or email test@example.com. <p>HTML tags here</p>  "
print(f"\nOriginal text:\n{repr(sample_text)}")

# Example 1: Lowercase conversion
print("\n--- Example 1: Lowercase ---")
result = TextCleaning(sample_text).lowercase().get()
print(f"Result: {repr(result)}")

# Example 2: Uppercase conversion
print("\n--- Example 2: Uppercase ---")
result = TextCleaning(sample_text).uppercase().get()
print(f"Result: {repr(result)}")

# Example 3: Remove punctuation
print("\n--- Example 3: Remove Punctuation ---")
result = TextCleaning(sample_text).remove_punctuation().get()
print(f"Result: {repr(result)}")

# Example 4: Remove punctuation but keep some
print("\n--- Example 4: Remove Punctuation (Keep some) ---")
result = TextCleaning(sample_text).remove_punctuation(keep=".,!?").get()
print(f"Result: {repr(result)}")

# Example 5: Expand contractions
print("\n--- Example 5: Expand Contractions ---")
result = TextCleaning(sample_text).expand_contractions().get()
print(f"Result: {repr(result)}")

# Example 6: Remove URLs and emails
print("\n--- Example 6: Remove URLs and Emails ---")
result = TextCleaning(sample_text).remove_urls().remove_emails().get()
print(f"Result: {repr(result)}")

# Example 7: Remove HTML tags
print("\n--- Example 7: Remove HTML Tags ---")
result = TextCleaning(sample_text).remove_html_tags().get()
print(f"Result: {repr(result)}")

# Example 8: Remove extra whitespace
print("\n--- Example 8: Remove Extra Whitespace ---")
result = TextCleaning(sample_text).remove_whitespace(mode="extra").get()
print(f"Result: {repr(result)}")

# Example 9: Chaining multiple operations
print("\n--- Example 9: Chaining Multiple Operations ---")
result = (TextCleaning(sample_text)
          .lowercase()
          .expand_contractions()
          .remove_urls()
          .remove_emails()
          .remove_html_tags()
          .remove_punctuation()
          .remove_whitespace(mode="extra")
          .get())
print(f"Result: {repr(result)}")

# Example 10: Split text
print("\n--- Example 10: Split Text ---")
text = "apple,banana,orange,grape"
result = TextCleaning(text).split_text(delimiter=',')
print(f"Result: {result}")

# Example 11: Simple tokenization (no NLTK required)
print("\n--- Example 11: Simple Tokenization ---")
text = "This is a simple sentence for tokenization"
result = TextCleaning(text).tokenize(method="simple")
print(f"Result: {result}")

print("\nPART 2: DATAFRAME COLUMN EXAMPLES")

df = pd.DataFrame({
    'text': [
        "I can't believe it's working!",
        "Visit https://example.com for more INFO.",
        "  Extra   spaces   everywhere  ",
        "Remove <b>HTML</b> tags and email@test.com",
        "DON'T SHOUT AT ME!!!"
    ],
    'id': [1, 2, 3, 4, 5]
})

print("\nOriginal DataFrame:")
print(df)

# Example 12: Lowercase on DataFrame column
print("\n--- Example 12: Lowercase on DataFrame ---")
df_cleaned = TextCleaning(df, text_column='text').lowercase().get()
print(df_cleaned)

# Example 13: Multiple operations on DataFrame
print("\n--- Example 13: Multiple Operations on DataFrame ---")
df_cleaned = (TextCleaning(df, text_column='text')
              .lowercase()
              .expand_contractions()
              .remove_urls()
              .remove_emails()
              .remove_html_tags()
              .remove_punctuation()
              .remove_whitespace(mode="extra")
              .get())
print(df_cleaned)

# Example 14: Custom function processing
print("\n--- Example 14: Custom Function Processing ---")
def custom_cleanup(text):
    # Custom logic: replace numbers with placeholder
    import re
    return re.sub(r'\d+', '[NUM]', text)

df_test = pd.DataFrame({
    'text': ["I have 5 apples and 10 oranges", "Call me at 123-456-7890"]
})
df_cleaned = TextCleaning(df_test, text_column='text').process_text(custom_cleanup).get()
print(df_cleaned)

print("\nPART 3: NLTK-BASED FEATURES")

try:
    import nltk
    
    print("\nNOTE: These examples require NLTK data to be downloaded.")
    print("If you get errors, run: download_nltk_data()")
    print("Or manually: import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')")
    
    # Example 15: Word tokenization with NLTK
    print("\n--- Example 15: Word Tokenization (NLTK) ---")
    text = "Hello! How are you doing today? I'm doing great."
    try:
        result = TextCleaning(text).tokenize(method="word")
        print(f"Result: {result}")
    except LookupError as e:
        print(f"Error: {e}")
    
    # Example 16: Sentence tokenization
    print("\n--- Example 16: Sentence Tokenization ---")
    text = "Hello! How are you? I'm doing great. What about you?"
    try:
        result = TextCleaning(text).tokenize(method="sentence")
        print(f"Result: {result}")
    except LookupError as e:
        print(f"Error: {e}")
    
    # Example 17: Remove stopwords
    print("\n--- Example 17: Remove Stopwords (English) ---")
    text = "This is a sample sentence with some common words"
    try:
        result = TextCleaning(text).remove_stopwords(language="english").get()
        print(f"Result: {result}")
    except LookupError as e:
        print(f"Error: {e}")
    
    # Example 18: Porter Stemming
    print("\n--- Example 18: Porter Stemming ---")
    text = "running runs runner ran easily fairly"
    result = TextCleaning(text).stem(method="porter").get()
    print(f"Result: {result}")
    
    # Example 19: Snowball Stemming (supports multiple languages)
    print("\n--- Example 19: Snowball Stemming ---")
    text = "running runs runner ran easily fairly"
    result = TextCleaning(text).stem(method="snowball", language="english").get()
    print(f"Result: {result}")
    
    # Example 20: Lemmatization
    print("\n--- Example 20: Lemmatization (Noun) ---")
    text = "running runs runner ran easily fairly"
    try:
        result = TextCleaning(text).lemmatize(pos="n").get()
        print(f"Result: {result}")
    except LookupError as e:
        print(f"Error: {e}")
    
    # Example 21: Lemmatization (Verb)
    print("\n--- Example 21: Lemmatization (Verb) ---")
    text = "running runs runner ran"
    try:
        result = TextCleaning(text).lemmatize(pos="v").get()
        print(f"Result: {result}")
    except LookupError as e:
        print(f"Error: {e}")
    
    # Example 22: Complete pipeline with NLTK features
    print("\n--- Example 22: Complete NLP Pipeline ---")
    df_nlp = pd.DataFrame({
        'text': [
            "I'm running quickly through the forest!",
            "The cats were playing with their toys.",
            "She doesn't like swimming in cold water."
        ]
    })
    print("\nOriginal DataFrame:")
    print(df_nlp)
    
    try:
        df_nlp_cleaned = (TextCleaning(df_nlp, text_column='text')
                         .lowercase()
                         .expand_contractions()
                         .remove_punctuation()
                         .remove_stopwords(language="english")
                         .stem(method="porter")
                         .get())
        print("\nCleaned DataFrame (with stemming):")
        print(df_nlp_cleaned)
    except LookupError as e:
        print(f"Error: {e}")
        print("Please download NLTK data using: download_nltk_data()")

except ImportError:
    print("\nNLTK is not installed. Install it with: pip install nltk")
    print("Then run: download_nltk_data() to download required data")

print("\nPART 4: CONTRACTIONS DICTIONARY")

contractions = get_contractions_dict()
print(f"\nTotal contractions available: {len(contractions)}")
print("\nSample contractions:")
sample_keys = list(contractions.keys())[:10]
for key in sample_keys:
    print(f"  {key:15} -> {contractions[key]}")

# Example 23: Using custom contractions
print("\n--- Example 23: Custom Contractions ---")
custom_contractions = {
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to"
}
text = "I'm gonna wanna do this"
result = TextCleaning(text).expand_contractions(custom_contractions=custom_contractions).get()
print(f"Original: {text}")
print(f"Result: {result}")

print("\nPART 5: OPERATION LOGGING")

# Example 24: View operation log
print("\n--- Example 24: Operation Log ---")
cleaner = (TextCleaning("Hello World!")
           .lowercase()
           .remove_punctuation()
           .remove_whitespace(mode="extra"))
result = cleaner.get()
log = cleaner.get_log()

print(f"Result: {result}")
print("\nOperations performed:")
for i, op in enumerate(log, 1):
    print(f"  {i}. {op['operation']}: {op['details']}")

print("\nPART 6: REAL-WORLD EXAMPLE - SOCIAL MEDIA TEXT CLEANING")
social_media_df = pd.DataFrame({
    'post': [
        "OMG! I can't believe this! 😱 Check out https://bit.ly/example #amazing",
        "Don't @ me but I think <b>AI</b> is gonna change everything! Contact: info@company.com",
        "  WHY   ARE   WE   SHOUTING???  🔊",
        "It's 2024 and we're still dealing with this... SMH 🤦",
        "You won't believe what happened! Click here -> http://spam.com #clickbait"
    ],
    'timestamp': pd.date_range('2024-01-01', periods=5)
})

print("\nOriginal social media posts:")
print(social_media_df)

# Clean the posts
print("\n--- Cleaning Pipeline ---")
df_cleaned = (TextCleaning(social_media_df, text_column='post')
              .lowercase()
              .expand_contractions()
              .remove_urls()
              .remove_emails()
              .remove_html_tags()
              .remove_punctuation()
              .remove_whitespace(mode="extra")
              .get())

print("\nCleaned posts:")
print(df_cleaned)

print("\n" + "=" * 80)
print("DEMO COMPLETE!")
print("=" * 80)
print("\nKey Features Demonstrated:")
print("✓ Lowercase/Uppercase conversion")
print("✓ Contraction expansion (with custom dictionary)")
print("✓ Punctuation removal")
print("✓ URL and email removal")
print("✓ HTML tag removal")
print("✓ Whitespace cleaning")
print("✓ Text splitting and tokenization")
print("✓ Stopwords removal (NLTK)")
print("✓ Stemming (Porter, Snowball, Lancaster)")
print("✓ Lemmatization (NLTK)")
print("✓ DataFrame column processing")
print("✓ Method chaining")
print("✓ Custom function processing")
print("✓ Operation logging")
print("\n" + "=" * 80)
