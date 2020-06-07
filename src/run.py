import click

from evaluation.evaluator import ThaiEvaluator
from evaluation.strategies.oov_strategies import NoActionOOVStrategy
from evaluation.strategies.base import NoStructuredSourceStrategy

from datasets.datasets import DatasetLoader, tasks
from embeddings.embeddings import load_embedding
from cli_config import CLI_OOV_OPTION, CLI_SS_OPTION


@click.command()
@click.option("--oov", help='Strategy to handle OOV: letters or deepcut. If empty none of them is applied')
@click.option("--ss", help='Integrating structed sources: wn1, wn2, cn1, cn2. '
                         'If empty only word embeddings are evaluated')
@click.option("-f", "--filter", help='Filter not found words', is_flag=True)
@click.argument("model")
@click.argument("format")
def run(oov, ss, model, filter, format):
    OovStrategyCls = CLI_OOV_OPTION.get(oov, NoActionOOVStrategy)
    SSStrategyCls = CLI_SS_OPTION.get(ss, NoStructuredSourceStrategy)

    embeddings = load_embedding(model, format)

    evaluator = ThaiEvaluator(
        oov_strategy=OovStrategyCls(),
        structured_source_strategy=SSStrategyCls(),
    )

    for task in tasks:
        dataset = DatasetLoader(task)
        dataset_data = dataset.get_data()
        result, hm = evaluator.evaluate(embeddings, dataset_data, filter_not_found=filter)
        print(task)
        print(result, hm)



if __name__ == '__main__':
    run()

description = 'Evaluation tool for Thai distributional models'
'with option of integrating structured sources '
'(WordNet, ConceptNet'
