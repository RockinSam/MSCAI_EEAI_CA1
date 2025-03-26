from model.randomforest import RandomForest
from model.chained_multi_output import ChainedMultiOutput


def model_predict(data, df, name):
    
    results = []
    print(name)

    if name == 'RandomForest':
        model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    elif name == 'ChainedMultiOutput':
        model = ChainedMultiOutput(name, data)
    # elif name == 'Hierarchical':
    #     pass
        
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)


def model_evaluate(model, data):
    model.print_results(data)