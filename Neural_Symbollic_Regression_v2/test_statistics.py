from nsr import NeuralSRModel
from statistical_significance import StatisticalSignificance

model = NeuralSRModel(...)
model.fit(X_train,y_train)

stats = model.get_statistics()
results = stats.full_report()
print(results.summary)
print()
print(results.coefficient_table)
print()
print(results.shapiro)
print()
print(results.breusch_pagan)
print()
print(results.durbin_watson)
print()
print(results.vif_table)