# 📚 Book Metadata Collector (Google Books API)

This repository contains a Python script used for academic research at **Utrecht University** as part of a Master's thesis. The goal is to enrich a dataset of ~200,000 book quotes (with title and author) by retrieving metadata such as:

- 📅 `publishedDate`
- 📚 `categories` (genres)
- 🔢 `industryIdentifiers` (ISBNs)

The metadata is retrieved via the **[Google Books API](https://developers.google.com/books)** and stored locally in a CSV file.

---

## 🔧 How It Works

- Input: A list of book titles and authors.
- The script sends a single API call per book using the Google Books API.
- The response is parsed and stored locally.
- The resulting data is **not shared or published**, and is only used for offline, non-commercial academic analysis.

---

## 🧠 Use Case

> This tool is used to analyze linguistic and semantic patterns of quoted versus non-quoted sentences in literature. It is part of a non-commercial Master's thesis in the field of computational creativity.

---

## ✅ Attribution

This project uses the **Google Books API**.  
Attribution is provided as required by [Google Books branding guidelines](https://developers.google.com/books/branding):

![Powered by Google Books](https://developers.google.com/books/images/powered_by_books.png)

---

## 🔒 Disclaimer

This tool is run locally by a single user.  
No data is served to end users or displayed in any public-facing interface.  
All processed data is anonymized at the embedding level and is used strictly for academic purposes.

