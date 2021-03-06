# Clustering Analysis of the Iowa Gambling Task
>Github repo:[_here_](https://github.com/dockreg/ca4015_assignment_1)

> This project will explore the data from the Iowa Gambling task using unsupervised learning algorithms. The data used for this project can be found [_here_](http://doi.org/10.5334/jopd.ak). The resulting online jupyter book can be found [_here_](https://dockreg.github.io/ca4015_assignment_1/intro.html) and the pdf version [_here_](https://github.com/dockreg/ca4015_assignment_1/blob/main/_build/pdf/book.pdf)

## Table of Contents
* [General Info](#general-information)
* [Requirements](#requirements)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
<!-- * [License](#license) -->


## General Information
- This project aims to explore unsupervised learning techniques
- It will also provide experience using jupyter notebooks, git, markdown, and clustering analysis


## Requirements

- python
- pandas
- matplotlib
- jupyter-book
- numpy
- scipy
- sklearn
- array
- pyppeteer


## Usage
To build the book - ensure all requirements have been imported to the appropriate environment and then enter the following commands

```
jupyter-book build ca4015_assignment_1/
```

then followed by 

```
ghp-import -n -p -f ca4015_assignment_1/_build/html
```

This will build the online book in a seperate gh-pages branch

To build a pdf of this book, ensure all requirements are installed and then enter the following command:

```
jupyter-book build ca4015_assignment_1/ --builder pdfhtml
```


## Project Status
Project is: completed


## Room for Improvement

I would like to further investigate this data set and my methods around k-means clustering. I would like to gain some further insights around trends for healthy individuals vs those with prefrontal cortex health issues. As this was my first project using k-means I would like to delve further into its uses, and any limitations it may have as an unsupervised learning algorithm




```python

```
