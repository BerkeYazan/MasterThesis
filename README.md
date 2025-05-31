# Computational Analysis of Linguistic Creativity in Literary Quotes

[Refer to the **master** branch for the updated code.](https://github.com/BerkeYazan/MasterThesis/tree/master)

## Master's Thesis Research Project

A multi-dimensional computational framework for identifying and characterizing the patterns that distinguish extraordinary language from ordinary expression through analysis of popular literary quotes from Goodreads.

## Abstract

This thesis develops a novel computational approach to understand and analyze linguistic and semantic patterns that distinguish ordinary language from extraordinary language. We define ordinary language as expressions that are statistically likely within their context, whereas extraordinary language refers to linguistic artifacts perceived as creative, valuable, or striking, due to properties that may be either measurable or inherently subjective. 

To validate our methods and conduct comparative analysis, we use a dataset of popular, user-annotated book quotes from Goodreads, a sample of non-quoted spans from the same books, and benchmark NLP task datasets. These quotes, drawn from different periods, authors, and genres, represent a curated collection of linguistically valuable expressions that offer an ideal testbed for our methods due to their widespread appraisal and diversity, while the combined dataset enables us to establish meaningful contrast with ordinary language.

Our approach analyzes various dimensions of language, including semantic embeddings, information-theoretic measures, part-of-speech (POS) tagging, and sentiment analysis. Rather than theorizing about the human dimension, which inevitably leads to broader discussions of cultural context and artistic interpretation, we focus on the measurable properties of the texts themselves. Through this approach, we aim to bridge the gap between subjective appraisal and objective evaluation and contribute to ongoing research in computational creativity.

## Research Question

**Primary Research Question:**
*How can computational approaches identify, characterize, and differentiate the patterns of extraordinary language from ordinary expression through multi-dimensional analysis of literary quotes?*

**Research Objectives:**

1. **Develop a multi-dimensional analytical framework** that quantifies linguistic creativity through multiple computational lenses, such as semantic embeddings, information-theoretic measures, and syntactic structures

2. **Understand the distinctive semantic patterns** that differentiate extraordinary language from ordinary expression through baseline comparisons with both source-text and non-literary corpora, and identify potential typologies of creativity through multi-dimensional visualization

3. **Create interpretable insights** from the measurable properties of extraordinary language by examining cross-dimensional relationships and using explainable AI approaches that bridge computational analysis with humanities-based understanding

## Theoretical Framework

### The Classification Challenge of Language Value

There are nearly infinite letter combinations possible in human language, but only a vanishingly small fraction hold meaning or value. Even when meaning is achieved, the exceptional value found in poetry, literary quotations, or powerful metaphors remains rare. This presents a fundamental classification challenge of distinguishing signal from noise, grammatical correctness from genuine meaning, and commonplace language from creative expression.

### Limitations of Current Approaches

Recent advances in natural language processing have introduced new dimensions to this classification challenge. While transformer-based language models can generate syntactically and semantically coherent text, there remains a lack of transparent frameworks for understanding and reproducing the features that differentiate ordinary from extraordinary language. This gap between generative capabilities and analytical understanding underscores the importance of more sophisticated multidimensional approaches to linguistic creativity.

Traditional NLP approaches have predominantly focused on novelty as a proxy for creativity, using metrics like surprisal or semantic distance. However, these approaches cannot fully capture what makes language impactful - a sentence may be meaningless yet surprising, while another may carry significant weight using relatively ordinary vocabulary.

### Multi-dimensional Framework Development

Our research addresses these limitations by developing a framework that integrates:

- **Semantic Embeddings**: Contextual vector representations
- **Information-theoretic Measures**: Entropy and surprisal calculations
- **Syntactic Analysis**: Part-of-speech patterns and structural features
- **Figurative Language Detection**: Metaphor and irony identification
- **Affective Features**: Sentiment and emotional resonance
- **Stylistic Markers**: Punctuation patterns and authorial fingerprints

## Methodology

### Dataset Construction

**Primary Dataset: Goodreads Literary Quotes**
- 227,900 high-quality English quotes with engagement metrics
- User-annotated content representing widespread appraisal
- Diverse temporal, authorial, and genre representation
- Controlled for quality through engagement thresholds

**Comparative Datasets:**
- Non-quoted spans from the same literary works (controlling for authorial style)
- NLP benchmark datasets (establishing ordinary language baselines)
- Cross-genre and cross-temporal samples for robustness testing

### Computational Analysis Pipeline

1. **Text Preprocessing**: Unicode normalization, tokenization, linguistic annotation
2. **Feature Extraction**: Multi-dimensional linguistic feature computation
3. **Comparative Analysis**: Statistical differentiation between extraordinary and ordinary language
4. **Pattern Recognition**: Identification of creativity typologies through clustering
5. **Interpretability Analysis**: Explainable AI approaches for insight generation

### Technical Implementation

**Core Technologies:**
- Python ecosystem for data processing and analysis
- Transformer-based models for semantic embeddings
- Information-theoretic libraries for entropy/surprisal calculation
- NLP toolkits for syntactic and stylistic analysis
- Statistical modeling for pattern identification

**Analysis Framework:**
- Multi-dimensional feature space construction
- Dimensionality reduction and visualization
- Cross-dimensional relationship analysis
- Statistical significance testing
- Interpretability through feature importance

## Expected Contributions

### To Computational Linguistics
- **Novel Framework**: Multi-dimensional approach to linguistic creativity analysis
- **Methodological Innovation**: Integration of information theory, semantics, and style
- **Benchmark Creation**: Curated dataset for creativity research validation

### To Digital Humanities
- **Quantitative Literary Analysis**: Computational approaches to textual value
- **Canon Studies**: Data-driven insights into literary appreciation patterns
- **Cultural Analytics**: Measurable properties of widely valued language

### To Computational Creativity
- **Feature Identification**: Systematic characterization of creative language patterns
- **Interpretability**: Explainable frameworks for creativity assessment
- **Bridge Building**: Connection between subjective appraisal and objective measurement

## Research Significance

This research addresses a fundamental gap in computational creativity by moving beyond single-dimensional novelty metrics toward comprehensive multi-dimensional analysis. By focusing on measurable textual properties rather than cultural interpretation, we aim to develop frameworks that are both scientifically rigorous and practically applicable.

The use of literary quotes as a testbed provides several advantages:
- **Quality Control**: User engagement serves as quality validation
- **Diversity**: Spans multiple genres, periods, and authors
- **Brevity**: Short texts suitable for detailed computational analysis
- **Value Recognition**: Represents collectively acknowledged linguistic excellence


**Institution**: Utrecht University

**Keywords**: Computational Creativity, Natural Language Processing, Digital Humanities, Literary Analysis, Multi-dimensional Analysis
