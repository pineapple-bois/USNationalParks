# Biodiversity in the United States National Park system

---
This project follows from my EDA project [North American Birds of Prey](https://github.com/pineapple-bois/Biodiversity-in-National-Parks) which utilised an abridged version of `species.csv` provided by [Codecademy](https://www.codecademy.com)

----
#### There are two datasets from [Kaggle](https://www.kaggle.com/datasets/nationalparkservice/park-biodiversity?select=species.csv) kindly made public domain by the U.S. National Parks Service.

----
`parks.csv`

With Columns;
- Park Code - National Parks Service park code. 
- Park Name - Official Park Name
- State - US State(s) in which park is location
- Acres - Size of the park
- Latitude 
- Longitude

[Source](https://irma.nps.gov/NPSpecies/)

---
`species.csv`

With Columns; 
- Species ID - NPS park code
- Park Name - Park in which species appears
- Category - One of Mammal, Bird etc
- Order 
- Family
- Scientific Name - of form, Genus species
- Common Names
- Record Status - Usually approved
- Occurrence - Has species presence been confirmed?
- Nativeness - Is species native/invasive
- Abundance - Commonality of sightings
- Seasonality
- Conservation Status - by [IUCN](https://www.iucnredlist.org) convention

---
## Scope of the project

### Birds of Prey in the SW United States.

We define birds of prey/predatory birds as [Raptors](https://www.blm.gov/sites/default/files/documents/files/Morley-Nelson-Snake-River-Birds-Of-Prey_More-About-Raptors.pdf) 


Typically, in possession of;
&nbsp;
- Hooked beaks
- Sharp Talons
- Keen eyesight
- A hypercarnivous diet

----

### [wrangling.ipynb](https://github.com/pineapple-bois/Biodiversity-In-National-Parks-Real-World/blob/main/wrangling.ipynb)
- Data is cleaned and organised

### [analysis.ipynb](https://github.com/pineapple-bois/Biodiversity-In-National-Parks-Real-World/blob/main/analysis.ipynb)
- Organised data is analysed 

---
#### Technologies used:
```
Git Zsh
Jupyter Notebooks
Tableau
```
#### Libraries used:
```
pandas
matplotlib.pyplot
matplotlib.patches
plotly.express
seaborn
numpy
```
#### Techniques used:
```
DataFrame manipulation
Data wrangling using regex
Visualisations
```
#### Images:

- Images were sourced royalty free from [Pixabay](https://pixabay.com)
- Map of the USA from [Google Earth](https://earth.google.com/web/@39.00737915,-95.31864374,-81.61621475a,5326276.02988026d,35y,0h,0t,0r)

#### References:

- [Wikipedia](https://en.wikipedia.org/wiki/Bird_of_prey)
- [US National Parks Service](https://www.nps.gov/index.htm)

----

