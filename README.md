# Global Video Game Genre Perdictions.

## Problem Statement
- This dataset focuses on the selling of video games and the type of game it is. There are several ways to categorize a video game. The most often used would be the genre of the game(). Not every game can be played on every system, so it is also important to know what system the game is being purchased on as well. It is always a good idea to know what genre of game one prefers to play as everyone has a preference.
- What Genre are customers ordering the most? Join me as we find out.
- Thankfully a video games price is not affected by just the genre what are some other key factors? If you are looking into a game it is crucial to understand the genre of a game as to avoid any future dissatisfaction. 

## Data Dictionary
Below is a data dictionary to explain the meaning of each variable or field in the dataset.

| Column Name | Description |
|------------- |-------------|
| Rank         | Ranking of the game based on global sales. (Integer)|
| Name         | Name of the game. (String)                          |
| Platform     | Platform the game was released on. (String)         |
| Year         | Year the game was released. (Integer)               |
| Genre        | Genre of the game. (String)                         |
| Publisher    | Publisher of the game. (String)                     |
| NA_Sales     | Sales of the game in North America. (Float)         |
| EU_Sales     | Sales of the game in Europe. (Float)                |
| JP_Sales     | Sales of the game in Japan. (Float)                 |
| Other_Sales  | Sales of the game in other regions. (Float)         |
| Global_Sales | Total sales of the game worldwide. (Float)          |
| ...          | ...                                                 |

## Executive Summary
There are 12 different Genres covered in this dataset with non of them overlapping one another. A game can have more than one genre yet this data set is focused on the genre that is most prevelant to the game.

### Data Cleaning Steps
I removed 'Rank' variable from the dataset as it would not help with the analysis in any way. I then had to turn the 'Genre' variable into a float for modeling. The numbers corresponding to the genre type are as follows.

0-Action
1-Sports
2-Misc
3-Role-Playing
4-Shooter
5-Adventure
6-Racing
7-Platform
8-Simulation
9-Fighting
10-Strategy
11-Puzzle

### Key Visualizations
Include key visualizations that highlight important aspects of the data. Use graphs, charts, or any other visual representation to make your points.

#### Visualization 1: Count of each Genre of Games
The countplot below shows that the Genre of game that is sold the most is Action based games where as the one selling the least is Strategy.

![NGenre](https://github.com/user-attachments/assets/7e114398-aa3b-4849-8b4c-a64252a4a493)

#### Visualization 2: North American Sales vs Global Sales
The scatterplot below shows that North American Sales have a significant affect on Global Sales with a correlation of 0.94.

![NA vs GS](https://github.com/user-attachments/assets/56d0e0b0-683f-42b3-8a41-51f139311b30)

## Model Performance

### Model Selection
I used KNN for this dataset. This was so I could use the classifier to get a more accurtate prediction.
### Evaluation Metrics
Summarize the performance of the model(s) using key evaluation metrics (e.g., RMSE, R²).

| Model             | RMSE     | R²       |
|-------------------|----------|----------|
| KNN               | [22.3%]  | [4.2]    |

## Conclusions/Recommendations
I will keep my recomendation short and sweet with this one by saying that if a puzzle game is taken and then has action added as the secondary genre you would sell more puzzle games. Mixing and matching the genres when creating a game can and will affect how often the game sells and for how much it can be sold for.

---
