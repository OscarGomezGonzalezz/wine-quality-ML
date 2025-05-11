# Wine Quality Data Mining #

We used Kaggle’s Red/White Wine Quality [dataset] to build various classification models to predict whether a particular wine is “good quality” or not. Each wine in this dataset is given a “quality” score between 0 and 10. For the purpose of this project, we converted the output to a binary output where each wine is either “good quality” (a score of 6.5 or higher) or not (a score below 6.5). The quality of a wine is determined by 11 input variables:

[dataset]: https://archive.ics.uci.edu/dataset/186/wine+quality


# Dataset Explanation

## Features (Inputs):
### fixed_acidity (Continuous)

Refers to non-volatile acids that do not evaporate easily. Common fixed acids in wine include tartaric, malic, and citric acids. They contribute to the overall acidity and structure of the wine.

### volatile_acidity (Continuous)

Represents the amount of acetic acid (main component of vinegar). Higher values can lead to an unpleasant vinegar taste and are generally considered a defect in wine.

### citric_acid (Continuous)

A naturally occurring acid in wine. It adds freshness and flavor, though it's present in smaller amounts compared to other acids. Excessive citric acid can make the wine taste sour.

### residual_sugar (Continuous)

Refers to the amount of sugar left in the wine after fermentation. Wines with higher residual sugar are sweeter. Dry wines have low residual sugar.

### chlorides (Continuous)

Indicates the salt content in the wine, mostly sodium chloride. High levels can indicate poor wine quality and may affect taste.

### free_sulfur_dioxide (Continuous)

The free form of SO₂ that acts as an antioxidant and antimicrobial agent. Helps in preserving the wine but must be balanced as too much can cause a pungent smell.

### total_sulfur_dioxide (Continuous)

Sum of free and bound sulfur dioxide. High levels can cause unpleasant aromas and allergic reactions in sensitive individuals.

### density (Continuous)

The density of the wine, which is closely related to its sugar and alcohol content. Denser wines may indicate higher sugar levels.

### pH (Continuous)

A measure of acidity. Lower pH means higher acidity. Wine typically has a pH between 3 and 4. Affects taste, color, and microbial stability.

### sulphates (Continuous)

Sulfate salts that can contribute to wine’s preservation and astringency (dryness). Also used as an antioxidant and antimicrobial agent.

### alcohol (Continuous)

The percentage of alcohol by volume. Higher alcohol content often correlates with better perceived quality in wine.

### Target (Output):
quality (Integer, Target Variable)

A score (usually given by wine tasters) ranging from 0 to 10 that reflects the perceived quality of the wine. This is the main variable you're likely trying to predict.