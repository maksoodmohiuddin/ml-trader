Q Learning Based Trader
To build the Q Learner based trader, I have used 3 technical indicators, Simple Moving Average, Bollinger Bands and Momentum, as my states (discretized), daily return value as my reward and three actions: 0, 1, & 2 representing Do Nothing, Go Long & Go Short respectively.
To setup the Q Leaner, in my Strategy Learner, I first discretized 3 technical indicators, Simple Moving Average, Bollinger Bands and Momentum, as my states using binning technique without any API. For training data for the learner, I use the provided symbols, ML4T-220, IBM, UNH, SINE_FAST_NOISE with in sample period of January 1, 2006 to December 31 2009 using 40 days as window or look-back period. Since I use 10 as steps for discretization and 3 actions, I initialize the Q table with 10 ^ 3 = 1000 states. For reward, I initialize the Q Table with very small value ranging from -0.001 to 0.001. After the initialization, I train Q Leaner to build a policy that will yield optimal actions based on its state. To do that during the learning process, I go through each trading day of in-sample training data and use daily return [position * daily return] as my reward for the action taken by Q learner for a state. For convergence, I simply run the leaner X (10) times and mark it as converged.  To satisfy the holding requirements [0, 500, -500], I follow the below trading strategy:  
If position = 0 and action is GO LONG:
Execute LONG 500  [Position 500]
else if position = 0 and action is GO SHORT:
	Execute SHORT 500 [Position -500]
else if position = 500 and action is GO SHORT:
	Execute SHORT 1000 [Position -500]
else if position = -500 and action is GO LONG:
	Execute LONG 1000 [Position 500]
else:
	Do Nothing

Now, I test the trained Q Leaner using Test Policy for 5 scenarios. The resulting performance are presented in the following table:
	Q Learner Based Trading 	Benchmark
MLT4_220 (In Sample) 	906.68%	25.72%
MLT4_220 (Out of Sample) 	210.80%	8.16%
IBM (In Sample) 	183.61%	25.72%
UNH (In Sample) 	109.59%	25.72%
SINE_FAST_NOISE (In Sample) 	1687.11%	25.72%

Note, that, I have used following as Benchmark: The performance of a portfolio starting with $100,000 cash, then investing in 500 shares of IBM in start date and holding that position and sell it the end date.
As indicated in the above table, Q Learner based trading out performs the benchmark substantially in all 5 scenarios. I think the reason behind that is Q Learner is a reinforcement-based learner so the learner learns to make actions that will only yield positive results, avoiding mistakes. This specific property of reinforcement-based learner makes it a great for trading since the Q Learner based trader will ONLY takes actions that will yield positive trade result. A note of caution though, underlying strength of Q Learner is still the technical indicators which may not always yield positive results. Technical indicator is fallible in certain market scenarios (for example, on December 12, 2016 a tweet by President Elect Trump lowered Lockheed Martin stock by 4%, which will be difficult to predict using technical indicators).  Nevertheless, Q Learner is a great example of how Machine Learning can be used to beat the market. 
