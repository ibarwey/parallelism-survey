from datasets import Metric, Value, Features, MetricInfo

def simple_acc(preds, labels):
    return (preds == labels).sum() / len(preds)

class ACCURACY(Metric):
    def _info(self):
        return MetricInfo(
            description="Calculates Accuracy metric.",
            citation="TODO: _CITATION",
            inputs_description="_KWARGS_DESCRIPTION",
            features=Features({
                'predictions': Value('int64'),
                'references': Value('int64'),
            }),
            format='numpy'
        )

    def _compute(self, predictions, references):
        return {"ACCURACY": simple_acc(predictions, references)}
