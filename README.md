# Credit card fraud detection using ML
* Name - V Sai Vivek 
* Project - Credit Card Fraud Detection 
* Skills - Logistic regression , Support Vector Machine, K Nearest Neighbours, F1 Score, ROC-AUC Curve, Data Visualisation , Exploratory Data Analysis , Data Science application in Finance , Machine Learning 
* Tools - Google Colab , Jupyter Notebooks , Python , Numpy , Pandas , Matplotlib , Seaborn , Sklearn
## Code :

The code is availaible in this repository itself.

## The Dataset :

The data was taken from Kaggle site : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud .

The Columns do not have physical significance directly visible since as per the source (Kaggle) , the data was compressed using Principle Component Analysis (PCA) in order to protect the privacy of the individuals while making a realistic scenario dataset availaible to public.

## Data Preprocessing and Visualisation :

![Screenshot 2025-04-07 214209](https://github.com/user-attachments/assets/290be9ae-cc16-41f2-8120-30fc8a84c57d)

### Correlations :

![output](https://github.com/user-attachments/assets/375dbc85-b917-49b3-9764-e3550e9a2cf6)

The columns do not seem to have correlations with each other , and seem to have great correlation with the Class and time variables , hence being a great indicator that simple models would be helpful here , and neural networks won't be needed.

### Relation between target variables and columns :

![image](https://github.com/user-attachments/assets/5614902d-4274-43f6-b383-4d2b856d3bd3)
![image](https://github.com/user-attachments/assets/f26f3460-4cfc-4a6e-95ad-be5f97ec9fb6)
![image](https://github.com/user-attachments/assets/c5358070-2582-4e78-b3a6-1356b559dccd)
![image](https://github.com/user-attachments/assets/4b684bd2-8418-4fca-bb53-726d190572dd)
![image](https://github.com/user-attachments/assets/d229ef0c-8503-42ca-8498-b41c0ab9b069)
![image](https://github.com/user-attachments/assets/e3d0f552-73c4-428c-8d18-10690dc35f36)
![image](https://github.com/user-attachments/assets/23f2a310-d051-473d-9db5-6eda5dfda36e)
![image](https://github.com/user-attachments/assets/974aedf4-623d-4b41-bff2-e9728a74adb0)
![image](https://github.com/user-attachments/assets/54a8baed-0831-4d6e-bf6a-c6e20edb7f1f)
![image](https://github.com/user-attachments/assets/6553cf1a-4668-45e9-b4a7-23cbf3c7110c)
![image](https://github.com/user-attachments/assets/01bd1346-8515-4069-9a0f-1447ddcb677b)
![image](https://github.com/user-attachments/assets/0b89b731-54c6-46bf-b2f1-daa618f256a9)
![image](https://github.com/user-attachments/assets/8e66d4fa-59e6-4759-8ff0-9d0fb55dd99a)
![image](https://github.com/user-attachments/assets/8233f919-7ce4-4c40-925e-01c54162a676)
![image](https://github.com/user-attachments/assets/37945c38-f77d-4133-96ce-05f93d63bfae)
![image](https://github.com/user-attachments/assets/7ef9557d-daab-4dde-b9f0-dfef7f0f6770)
![image](https://github.com/user-attachments/assets/c25d19e2-4706-41bf-96f5-b9c16ee6d39a)
![image](https://github.com/user-attachments/assets/766de750-6e96-43df-a546-861eb80851ef)
![image](https://github.com/user-attachments/assets/8dfc40d0-6f5d-4a78-96bb-522f7aafdb2b)
![image](https://github.com/user-attachments/assets/b4210706-70cf-4228-9192-4b08885add32)
![image](https://github.com/user-attachments/assets/e0e54869-ec67-4f8a-bff8-8ef84b9e68a7)

A plot between different columns and amount along with different colours for target variable show that our output classes are separable by linear boundary even in case of graphing variables alone , hence __LOGISTIC REGRESSION__ will help separate the multivariable data into 2 classes.

### Class Imbalance in dataset :

![image](https://github.com/user-attachments/assets/bff44d48-2efe-4ac5-8631-3b3c85b00488)

This shows that we have way way less data for fraud cases than for non fraud cases , which is expected from the dataset. 

To cure imbalance , we can use undersampling or oversampling. Here, I have decided to use SMOTE to counter the class imbalance in the dataset. 

### Training the model :

I have trained a Logistic Regression Model here. The model was showing a not converging warning , so I read its documentation and added the code to make it run for 150 iterations.

### Results from part 1 :

![Screenshot 2025-04-07 215806](https://github.com/user-attachments/assets/0197b1f4-58cb-4027-b14b-0c726aaa9b27)

The F1 score came 0.99 meaning the Classifier is working great. It managed to catch 91 out of 101 frauds , thus preventing frauds 90% of the time. 
The confusion matrix , precision , recall and F1 score has been displayed for your convenience. The confusion matrix readings and the F1 show the success of the project.

### Results from part 2 :

I have also uploaded some raw code to this repository , here are the conclusions derived from it :

Frauds are time independent so we can drop time :

![image](https://github.com/user-attachments/assets/3f0ac5a2-105f-460f-b9a7-7cc837c6f77a)

I decided to __undersample__ the dataset since significance of the data would be more realistic if there was no synthetic dataset. 
I also decided to choose the ML model with most recall , reason being that I realized later that as a business, labelling a Non Fraud datapoint as fraudulent would be much more worse for the company, __since nobody would like their card to decline__ and people would literally stop using that credit card, so we must __focus more on achieving lower recall than only blindly improving F1 score__ . 
So I got Logistic regression as the winner again with the following results :

![Screenshot 2025-04-07 220836](https://github.com/user-attachments/assets/860518bd-a6bc-49e9-87ee-767e213ac28d)

Other models weren't much far behind regarding performance too , but I decided to keep the final code clean and keep the trial and error part in the "raw_code" file . 












































