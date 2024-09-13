from diagrams import Diagram, Cluster
from diagrams.generic.compute import Rack
from diagrams.generic.storage import Storage
from diagrams.onprem.client import Users
from diagrams.programming.language import Python

with Diagram("Random Forest Workflow", show=False, direction="TB"):
    # Input Dataset from User
    user = Users("Load Dataset")
    dataset = Storage("Dataset (Features & Target)")

    # Split the Dataset
    with Cluster("Split Dataset"):
        training_set = Storage("Training Set")
        test_set = Storage("Test Set")

    user >> dataset >> [training_set, test_set]

    # Random Forest Initialization
    rf_init = Rack("Random Forest Initialization")

    # Bootstrap Sampling and Building Trees
    with Cluster("Bootstrap Sampling & Tree Building"):
        with Cluster("Tree 1"):
            tree1_sample = Storage("Bootstrap Sample 1")
            tree1 = Rack("Decision Tree 1")

        with Cluster("Tree 2"):
            tree2_sample = Storage("Bootstrap Sample 2")
            tree2 = Rack("Decision Tree 2")

        with Cluster("Tree 3"):
            tree3_sample = Storage("Bootstrap Sample 3")
            tree3 = Rack("Decision Tree 3")

    # Best Split Selection and Recursive Tree Growth
    with Cluster("Recursive Tree Growth"):
        split_selection1 = Rack("Best Split Tree 1")
        split_selection2 = Rack("Best Split Tree 2")
        split_selection3 = Rack("Best Split Tree 3")

    rf_init >> [tree1_sample, tree2_sample, tree3_sample]
    tree1_sample >> tree1 >> split_selection1
    tree2_sample >> tree2 >> split_selection2
    tree3_sample >> tree3 >> split_selection3

    # Aggregate Predictions
    with Cluster("Prediction and Aggregation"):
        tree_preds = [tree1, tree2, tree3]
        prediction = Python("Aggregate Predictions")

    split_selection1 >> tree1 >> prediction
    split_selection2 >> tree2 >> prediction
    split_selection3 >> tree3 >> prediction

    # Evaluation
    evaluation = Rack("Evaluate Performance (MSE)")
    prediction >> evaluation