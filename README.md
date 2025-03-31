# Text Clustering Methods

This is Msc. Thesis on the subject of comparing ML methods to cluster textual data with advancements of LLMs. 

## Datasets
3 datasets are used to measure how well clustering works:
1) 20 News Groups. 
This dataset is available from sklearn.datasets.

2) BBC News: 5 groups of texts
To download this dataset use the following link: https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive

3) AG News: 4 groups of news - "world", "sports", "business", and "Science"
To download this dataset use the following link: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset?resource=download

## Methods

## Noise

In text clustering, noise can be introduced to test how robust the clustering algorithm is. Below are different types of noise applied to the same text sample.

### Adding No Noise

```
"Hey, I am a bunch of texts.\nI like to be grouped with similar texts.\nIf you put me in the wrong group, I'll feel lost.\nSo, find a way to cluster me with others like me!"
```

---

### Adding Random Characters Noise

```
"Hey, I am a bunch of texts. I like to be grouped with similar texts. If you put me in the wrong group, I'll feel lost. So, find a way to cluster me with others like me!"
```

---

### Adding Random Words Noise

```
"Hey, I am a bunch of texts. I like hausmannite to be grouped decide with similar texts. If you put me in the wrong prepossessing group, I'll feel lost. So, find a way to cluster me with others like me!"
```

---

### Deleting Random Words Noise

```
"Hey, I am a bunch of texts. I like to be with similar texts. If you put me in the wrong group, I'll feel lost. So, find way to cluster me with others me!"
```

---

### Shuffling Sentences Noise

```
"I like to be grouped with similar texts. Hey, I am a bunch of texts. If you put me in the wrong group, I'll feel lost. So, find a way to cluster me with others like me!"
```

---

### Replacing with Synonyms Noise

```
"Hey , I am a lot of texts . I like to be grouped with similar texts . If you put me in the wrong group , I 'll feel lost . So , find a way to cluster me with others like me !"
```

---

### Replacing with Antonyms Noise

```
"Hey , I am a bunch of texts . I like to be grouped with similar texts . If you put me in the wrong group , I 'll feel lost . So , find a way to cluster me with others like me !"
```

