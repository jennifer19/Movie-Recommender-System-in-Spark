Author: Prakhar Dogra
G Number: G01009586
CS 657 Assignment 4

The following README gives details about the files contained in this folder:

1. Dataset
	The dataset was downloaded from the MovieLens website from the following link : http://grouplens.org/datasets/movielens/

2. SourceCode
	This folder contains the Source Code of the recommender system used to complete the tasks of the assignment: RecommenderSystem.py

3. PseudoCode
	This folder contains the Pseudo Code for Cross Validation and making recommendations for new user added to database: RecommenderSystem.pdf

4. Output
	Average Mean Squared Error = 0.7113683147263539                                                                                                         
	Average Root Mean Squared Error = 0.8434265319079984                                                                                                    
	Average Mean Average Precision = 0.7320988264859689 
	Recommending 5 movies:
	[('Offside (2006)', 4.110878897163405), ('Lonely Are the Brave (1962)', 4.298455148020707), ('Survive and Advance (2013)', 4.772034144517072), 
	('Seven Chances (1925)', 4.458627960405586), ('Yojimbo (1961)', 4.268382571411703)]
	
5. General Information
	- Initially all the programs were executed to perform Cross Validation and then the code, to recommend movies to the new user added to the database, was added at the end.
	- Cross Validation was done on parameters: rank and lambda_
	