# Submission 1: ML Pipeline - Intent Classification on ATIS Airline

Username Dicoding: nna_alif

![](https://miro.medium.com/v2/resize:fit:1400/1*Xe8qYW2BdcWc1U5PRCgoXw.png)


| Scope Information | Description |
| --- | --- |
| Dataset | [ATIS Airline Travel Information System](https://www.kaggle.com/datasets/hassanamin/atis-airlinetravelinformationsystem/data?select=atis_intents.csv) |
| Problem Statement | The ATIS dataset, a benchmark in intent classification, comprises messages and associated intents from the Airline Travel Information System, serving as a valuable resource for training classifiers in Natural Language Understanding (NLU) systems within chatbots. Accurately discerning user intent is crucial for enhancing the effectiveness of chatbot interactions, making the ATIS dataset an invaluable asset for developing and refining intent classification models in chatbot platforms. |
| ML Solution | Create machine learning solution to determine intent context is about Flight (1) / Not Flight Contenxt (0) to easily segment message from user. |
| Processing Methods | Using `TextVectorization()` for word embedding |
| Model Architecture | Using `LSTM()` as main preprocessing hidden layer, also include `Dense()` and `Dropout()` layers to enhance weight transfer to determine message is Flight (1) / Not Flight Context (0) using `Sigmoid` activation and evaluate model using `BinaryCrossEntropy`|
| Model Performance | according `Binary Accuracy` metrics to evaluate training set 99.36% and evaluation set 98.44% to determine message is 1 / 0 already show us, the LSTM architecture is already perform the best situation. |