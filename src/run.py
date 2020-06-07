from multiprocessing import Process

import click

from evaluation.evaluator import ThaiEvaluator
from evaluation.strategies.oov_strategies import NoActionOOVStrategy
from evaluation.strategies.base import NoStructuredSourceStrategy

from datasets.datasets import DatasetLoader, tasks
from embeddings.embeddings import load_embedding
from cli_config import CLI_OOV_OPTION, CLI_SS_OPTION


def _process_work(evaluator, task, embeddings, f):
    dataset = DatasetLoader(task)
    dataset_data = dataset.get_data()
    result, hm = evaluator.evaluate(embeddings, dataset_data, filter_not_found=f)
    print(task)
    print(result, hm)


@click.command()
@click.option("--oov", help='Strategy to handle OOV: letters or deepcut. If empty none of them is applied')
@click.option("--ss", help='Integrating structed sources: wn1, wn2, cn1, cn2. '
                           'If empty only word embeddings are evaluated')
@click.option("-f", help='Filter not found words', is_flag=True)
@click.argument("model")
@click.argument("format")
def run(oov, ss, f, model, format):
    OovStrategyCls = CLI_OOV_OPTION.get(oov, NoActionOOVStrategy)
    SSStrategyCls = CLI_SS_OPTION.get(ss, NoStructuredSourceStrategy)

    embeddings = load_embedding(model, format)

    evaluator = ThaiEvaluator(
        oov_strategy=OovStrategyCls(),
        structured_source_strategy=SSStrategyCls(),
    )

    processes = []
    for task in tasks:
        process = Process(target=_process_work, args=(evaluator, task, embeddings, f))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


if __name__ == '__main__':
    run()

description = 'Evaluation tool for Thai distributional models'
'with option of integrating structured sources '
'(WordNet, ConceptNet'
