import re
import string
from typing import List, Union, Dict, Callable, Optional
import pandas as pd

# Try importing NLTK resources, but make them optional
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Some features will be limited. Install with: pip install nltk")

CONTRACTIONS_DICT = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because",
    "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not", 
    "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
    "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is", 
    "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "i'd": "i would", "i'd've": "i would have",
    "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", 
    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
    "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", 
    "shouldn't've": "should not have", "so've": "so have", "so's": "so is", "that'd": "that would", "that'd've": "that would have",
    "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "they'd": "they would",
    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are",
    "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
    "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
    "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have",
    "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have",
    "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is",
    "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", 
    "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
    "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"
}

# ----------------- TEXT DATA CLEANING -----------------
class TextCleaning:
    """
    A class for cleaning and preprocessing text data.
    Supports both single text strings and pandas DataFrame columns.
    
    Usage:
        # For single text
        cleaner = TextCleaning("Hello World!")
        result = cleaner.lowercase().remove_punctuation().get()
        
        # For DataFrame columns
        df = pd.DataFrame({'text': ['Hello!', 'World!']})
        cleaner = TextCleaning(df, text_column='text')
        df_cleaned = cleaner.lowercase().remove_punctuation().get()
    """
    
    def __init__(self, data: Union[str, pd.DataFrame], text_column: Optional[str] = None, copy: bool = True):
        """
        Initialize TextCleaning with either a string or a DataFrame.
        
        Args:
            data: Either a string or a pandas DataFrame
            text_column: Column name if data is a DataFrame. Required for DataFrames.
            copy: Whether to create a copy of the DataFrame (if applicable)
        """
        self.is_dataframe = isinstance(data, pd.DataFrame)
        self.operations_log = []
        
        if self.is_dataframe:
            if text_column is None:
                raise ValueError("text_column must be specified when using a DataFrame")
            if text_column not in data.columns:
                raise KeyError(f"Column '{text_column}' not found in DataFrame")
            self.df = data.copy() if copy else data
            self.text_column = text_column
            self.original_shape = data.shape
        else:
            if not isinstance(data, str):
                raise TypeError("Data must be either a string or a pandas DataFrame")
            self.text = data
            self.text_column = None
    
    def _log_operation(self, operation: str, details: str) -> None:
        """Log operations for debugging and tracking"""
        self.operations_log.append({
            'operation': operation,
            'details': details,
            'timestamp': pd.Timestamp.now() if pd else None
        })
    
    def _apply_to_text(self, func: Callable[[str], str]) -> 'TextCleaning':
        """Apply a function to text, whether it's a string or DataFrame column"""
        if self.is_dataframe:
            self.df[self.text_column] = self.df[self.text_column].astype(str).apply(func)
        else:
            self.text = func(self.text)
        return self
    
    def lowercase(self) -> 'TextCleaning':
        """Convert text to lowercase."""
        self._apply_to_text(lambda x: x.lower())
        self._log_operation("lowercase", "Converted text to lowercase")
        return self
    
    def uppercase(self) -> 'TextCleaning':
        """Convert text to uppercase."""
        self._apply_to_text(lambda x: x.upper())
        self._log_operation("uppercase", "Converted text to uppercase")
        return self
    
    def titlecase(self) -> 'TextCleaning':
        """Convert text to title case (first letter of each word capitalized)."""
        self._apply_to_text(lambda x: x.title())
        self._log_operation("titlecase", "Converted text to title case")
        return self
    
    def expand_contractions(self, custom_contractions: Optional[Dict[str, str]] = None) -> 'TextCleaning':
        """
        Expand contractions in text (e.g., "don't" -> "do not").
        
        Args:
            custom_contractions: Optional dictionary of custom contractions to add/override
        """
        contractions = CONTRACTIONS_DICT.copy()
        if custom_contractions:
            contractions.update(custom_contractions)
        
        # Create regex pattern for contractions
        contractions_pattern = re.compile('(%s)' % '|'.join(contractions.keys()), flags=re.IGNORECASE|re.DOTALL)
        
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded = contractions.get(match.lower(), match)
            # Preserve original capitalization
            if first_char.isupper():
                expanded = expanded.capitalize()
            return expanded
        
        def expand_text(text: str) -> str:
            return contractions_pattern.sub(expand_match, text)
        
        self._apply_to_text(expand_text)
        self._log_operation("expand_contractions", f"Expanded contractions using {len(contractions)} patterns")
        return self
    
    def remove_punctuation(self, keep: Optional[str] = None) -> 'TextCleaning':
        """
        Remove punctuation from text.
        
        Args:
            keep: Optional string of punctuation characters to keep (e.g., ".,!?")
        """
        if keep is None:
            punct_to_remove = string.punctuation
        else:
            punct_to_remove = ''.join([p for p in string.punctuation if p not in keep])
        
        translator = str.maketrans('', '', punct_to_remove)
        
        self._apply_to_text(lambda x: x.translate(translator))
        self._log_operation("remove_punctuation", f"Removed punctuation (kept: '{keep if keep else 'none'}')")
        return self
    
    def remove_digits(self) -> 'TextCleaning':
        """Remove all digits from text."""
        self._apply_to_text(lambda x: re.sub(r'\d+', '', x))
        self._log_operation("remove_digits", "Removed all digits")
        return self
    
    def remove_whitespace(self, mode: str = "extra") -> 'TextCleaning':
        """
        Remove whitespace from text.
        
        Args:
            mode: "extra" (remove extra spaces), "all" (remove all spaces), "leading_trailing" (strip only)
        """
        def clean_whitespace(text: str) -> str:
            if mode == "extra":
                return ' '.join(text.split())
            elif mode == "all":
                return text.replace(' ', '')
            elif mode == "leading_trailing":
                return text.strip()
            else:
                raise ValueError(f"Unknown mode '{mode}'. Use 'extra', 'all', or 'leading_trailing'")
        
        self._apply_to_text(clean_whitespace)
        self._log_operation("remove_whitespace", f"Removed whitespace using mode '{mode}'")
        return self
    
    def remove_urls(self) -> 'TextCleaning':
        """Remove URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        self._apply_to_text(lambda x: re.sub(url_pattern, '', x))
        self._log_operation("remove_urls", "Removed URLs")
        return self
    
    def remove_emails(self) -> 'TextCleaning':
        """Remove email addresses from text."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self._apply_to_text(lambda x: re.sub(email_pattern, '', x))
        self._log_operation("remove_emails", "Removed email addresses")
        return self
    
    def remove_html_tags(self) -> 'TextCleaning':
        """Remove HTML tags from text."""
        html_pattern = r'<[^>]+>'
        self._apply_to_text(lambda x: re.sub(html_pattern, '', x))
        self._log_operation("remove_html_tags", "Removed HTML tags")
        return self
    
    def split_text(self, delimiter: str = ' ', max_split: int = -1) -> Union[List[str], pd.DataFrame]:
        """
        Split text by delimiter.
        
        Args:
            delimiter: String delimiter to split on
            max_split: Maximum number of splits (-1 for unlimited)
        
        Returns:
            List of strings if input was string, DataFrame with split column if DataFrame
        """
        if self.is_dataframe:
            if max_split == -1:
                self.df[self.text_column + '_split'] = self.df[self.text_column].str.split(delimiter)
            else:
                self.df[self.text_column + '_split'] = self.df[self.text_column].str.split(delimiter, n=max_split)
            self._log_operation("split_text", f"Split text by '{delimiter}' (max_split={max_split})")
            return self
        else:
            if max_split == -1:
                result = self.text.split(delimiter)
            else:
                result = self.text.split(delimiter, max_split)
            self._log_operation("split_text", f"Split text by '{delimiter}' (max_split={max_split})")
            return result
    
    def tokenize(self, method: str = "word") -> Union[List[str], 'TextCleaning']:
        """
        Tokenize text into words or sentences.
        
        Args:
            method: "word" for word tokenization, "sentence" for sentence tokenization,
                   "simple" for simple whitespace-based word tokenization
        
        Returns:
            List of tokens if input was string, DataFrame with tokenized column if DataFrame
        """
        if method == "simple":
            # Simple tokenization (doesn't require NLTK)
            if self.is_dataframe:
                self.df[self.text_column + '_tokens'] = self.df[self.text_column].str.split()
                self._log_operation("tokenize", f"Tokenized text using simple method")
                return self
            else:
                result = self.text.split()
                self._log_operation("tokenize", f"Tokenized text using simple method")
                return result
        
        # NLTK-based tokenization
        if not NLTK_AVAILABLE:
            raise ImportError("NLTK is required for advanced tokenization. Install with: pip install nltk")
        
        try:
            if method == "word":
                tokenizer = word_tokenize
            elif method == "sentence":
                tokenizer = sent_tokenize
            else:
                raise ValueError(f"Unknown tokenization method '{method}'. Use 'word', 'sentence', or 'simple'")
            
            if self.is_dataframe:
                self.df[self.text_column + '_tokens'] = self.df[self.text_column].apply(tokenizer)
                self._log_operation("tokenize", f"Tokenized text using method '{method}'")
                return self
            else:
                result = tokenizer(self.text)
                self._log_operation("tokenize", f"Tokenized text using method '{method}'")
                return result
        except LookupError:
            raise LookupError(
                "NLTK data not found. Download with: "
                "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
            )
    
    def remove_stopwords(self, language: str = "english", custom_stopwords: Optional[List[str]] = None) -> 'TextCleaning':
        """
        Remove stopwords from text.
        
        Args:
            language: Language for stopwords. Options: "english", "spanish", "french", "german",
                     "italian", "portuguese", "dutch", "russian", "arabic", "indonesian", etc.
            custom_stopwords: Optional list of additional stopwords to remove
        
        Note: Requires NLTK. Available languages depend on NLTK's stopwords corpus.
        """
        if not NLTK_AVAILABLE:
            raise ImportError("NLTK is required for stopword removal. Install with: pip install nltk")
        
        try:
            stop_words = set(stopwords.words(language))
            if custom_stopwords:
                stop_words.update(custom_stopwords)
            
            def remove_stops(text: str) -> str:
                # Simple tokenization for stopword removal
                words = text.split()
                filtered_words = [word for word in words if word.lower() not in stop_words]
                return ' '.join(filtered_words)
            
            self._apply_to_text(remove_stops)
            self._log_operation("remove_stopwords", f"Removed stopwords for language '{language}'")
            return self
        except LookupError:
            raise LookupError(
                f"NLTK stopwords data not found. Download with: "
                f"import nltk; nltk.download('stopwords')"
            )
    
    def stem(self, method: str = "porter", language: str = "english") -> 'TextCleaning':
        """
        Apply stemming to reduce words to their root form.
        
        Args:
            method: Stemming algorithm. Options: "porter", "snowball", "lancaster"
            language: Language for Snowball stemmer (ignored for Porter and Lancaster)
                     Options: "english", "spanish", "french", "german", "italian", "portuguese",
                     "dutch", "swedish", "norwegian", "danish", "russian", "finnish", etc.
        
        Note: Requires NLTK. Porter is fastest but least accurate, Lancaster is most aggressive,
              Snowball is a good balance and supports multiple languages.
        """
        if not NLTK_AVAILABLE:
            raise ImportError("NLTK is required for stemming. Install with: pip install nltk")
        
        # Initialize stemmer based on method
        if method == "porter":
            stemmer = PorterStemmer()
        elif method == "snowball":
            try:
                stemmer = SnowballStemmer(language)
            except ValueError:
                available_langs = ", ".join(SnowballStemmer.languages)
                raise ValueError(f"Language '{language}' not supported. Available: {available_langs}")
        elif method == "lancaster":
            stemmer = LancasterStemmer()
        else:
            raise ValueError(f"Unknown stemming method '{method}'. Use 'porter', 'snowball', or 'lancaster'")
        
        def stem_text(text: str) -> str:
            words = text.split()
            stemmed_words = [stemmer.stem(word) for word in words]
            return ' '.join(stemmed_words)
        
        self._apply_to_text(stem_text)
        self._log_operation("stem", f"Applied stemming using {method} method (language: {language})")
        return self
    
    def lemmatize(self, pos: str = "n") -> 'TextCleaning':
        """
        Apply lemmatization to reduce words to their dictionary form.
        
        Args:
            pos: Part of speech. Options: "n" (noun), "v" (verb), "a" (adjective), "r" (adverb), "s" (satellite adjective)
        
        Note: Requires NLTK and WordNet. Lemmatization is more accurate than stemming but slower.
        """
        if not NLTK_AVAILABLE:
            raise ImportError("NLTK is required for lemmatization. Install with: pip install nltk")
        
        try:
            lemmatizer = WordNetLemmatizer()
            
            def lemmatize_text(text: str) -> str:
                words = text.split()
                lemmatized_words = [lemmatizer.lemmatize(word, pos=pos) for word in words]
                return ' '.join(lemmatized_words)
            
            self._apply_to_text(lemmatize_text)
            self._log_operation("lemmatize", f"Applied lemmatization with POS tag '{pos}'")
            return self
        except LookupError:
            raise LookupError(
                "NLTK WordNet data not found. Download with: "
                "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
            )
    
    def replace_pattern(self, pattern: str, replacement: str = "", regex: bool = True) -> 'TextCleaning':
        """
        Replace text matching a pattern with a replacement string.
        
        Args:
            pattern: Pattern to search for (regex if regex=True, else literal string)
            replacement: String to replace matches with
            regex: Whether to treat pattern as regex
        """
        if regex:
            self._apply_to_text(lambda x: re.sub(pattern, replacement, x))
        else:
            self._apply_to_text(lambda x: x.replace(pattern, replacement))
        
        self._log_operation("replace_pattern", f"Replaced pattern '{pattern}' with '{replacement}' (regex={regex})")
        return self
    
    def process_text(self, func: Callable[[str], str]) -> 'TextCleaning':
        """
        Apply a custom function to process text.
        
        Args:
            func: Function that takes a string and returns a processed string
        """
        self._apply_to_text(func)
        self._log_operation("process_text", "Applied custom function")
        return self
    
    def get(self) -> Union[str, pd.DataFrame]:
        """
        Get the processed text or DataFrame.
        
        Returns:
            Processed string if input was string, DataFrame if input was DataFrame
        """
        if self.is_dataframe:
            return self.df
        else:
            return self.text
    
    def get_log(self) -> List[Dict]:
        """
        Get the log of operations performed.
        
        Returns:
            List of operation logs
        """
        return self.operations_log


# Utility function for quick access to contractions dictionary
def get_contractions_dict() -> Dict[str, str]:
    """
    Get the built-in contractions dictionary.
    
    Returns:
        Dictionary mapping contractions to their expanded forms
    """
    return CONTRACTIONS_DICT.copy()


# Helper function to download NLTK data
def download_nltk_data():
    """
    Download required NLTK data packages.
    Run this function once to set up NLTK resources.
    """
    if not NLTK_AVAILABLE:
        print("NLTK is not installed. Install with: pip install nltk")
        return
    
    import nltk
    
    packages = [
        'punkt',           # For tokenization
        'punkt_tab',       # Additional tokenization data
        'stopwords',       # For stopword removal
        'wordnet',         # For lemmatization
        'omw-1.4',         # Open Multilingual Wordnet
    ]
    
    print("Downloading NLTK data packages...")
    for package in packages:
        try:
            nltk.download(package, quiet=True)
            print(f"✓ {package}")
        except Exception as e:
            print(f"✗ {package}: {str(e)}")
    print("Done!")
