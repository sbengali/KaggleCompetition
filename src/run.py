# load data
import train as tr
from sklearn.cross_validation import KFold

if __name__ == '__main__':
    train, encoder = tr.loadTrainSet()
    cv = KFold(train.shape[0], n_folds=8, shuffle=True)
    # Classify the data and pass the parameters to the training model
    train_model = tr.trainModel(tr.classification_pipe, train.ingredients, train.cuisine, cv, n_jobs=-1)

    test = tr.loadTestSet()
    test['cuisine'] = train_model.predict(test.ingredients)
    print test['cuisine']
    test[['id', 'cuisine']].to_csv("../output/prediction.csv", index=False)




