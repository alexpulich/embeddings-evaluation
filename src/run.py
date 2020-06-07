from multiprocessing import Manager, Process

import click

from evaluation.evaluator import ThaiEvaluator
from evaluation.strategies.oov_strategies import NoActionOOVStrategy
from evaluation.strategies.base import NoStructuredSourceStrategy

from datasets.datasets import DatasetLoader, tasks, DatasetEnum
from embeddings.embeddings import load_embedding
from cli_config import CLI_OOV_OPTION, CLI_SS_OPTION


def _print_report(results_dict):
    for task, result in results_dict.items():
        print(f'Task {task}')
        print(result)


def _print_latex_report(results_dict):
    latex1, latex2 = '', ''
    for task in tasks:
        result = results_dict[task]
        perc_oov_words = 100 * (
                result['num_missing_words'] / (result['num_found_words'] + float(result['num_missing_words'])))

        latex1 += '{:4.3f}~~{:4.3f}~~{:4.3f} & {:3.1f}~~{:3d}  & '.format(
            round(result['spearmanr'], 3),
            round(result['pearsonr'], 3),
            result['hm'],
            perc_oov_words,
            result['num_oov_word_pairs']
        )

        latex2 += '{:4.3f}~~{:4.3f}~~{:4.3f} & {:3.1f}~~{:3d}  & '.format(
            round(result['spearmanr'], 3),
            round(result['pearsonr'], 3),
            result['hm'],
            perc_oov_words,
            result['y.shape'][0]
        )

    print(latex1)
    print(latex2)


def _process_work(evaluator, task, embeddings, f, results_dict):
    dataset = DatasetLoader(task)
    dataset_data = dataset.get_data()
    result = evaluator.evaluate(embeddings, dataset_data, filter_not_found=f)
    results_dict[task] = result

    # TODO test for race condition and so on
    # print(task)
    # if not isinstance(evaluator._structured_source_strategy, NoStructuredSourceStrategy):
    #     print(f"Best alpha: {result.get('coef')}")


@click.command()
@click.option("--oov", help='Strategy to handle OOV: letters or deepcut. If empty none of them is applied')
@click.option("--ss", help='Integrating structed sources: wn1, wn2, cn1, cn2. '
                           'If empty only word embeddings are evaluated')
@click.option("-f", help='Filter not found words', is_flag=True)
@click.option("-m", "--multiprocess", help='Using multiprocessing to parallel datasets evaluation', is_flag=True)
@click.argument("model")
@click.argument("format")
def run(oov, ss, f, multiprocess, model, format):
    OovStrategyCls = CLI_OOV_OPTION.get(oov, NoActionOOVStrategy)
    SSStrategyCls = CLI_SS_OPTION.get(ss, NoStructuredSourceStrategy)

    embeddings = load_embedding(model, format)

    evaluator = ThaiEvaluator(
        oov_strategy=OovStrategyCls(),
        structured_source_strategy=SSStrategyCls(),
    )

    if multiprocess:
        processes = []
        manager = Manager()
        results_dict = manager.dict()
        for task in tasks:
            process = Process(target=_process_work, args=(evaluator, task, embeddings, f, results_dict))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
    else:
        results_dict = {}
        for task in tasks:
            _process_work(evaluator, task, embeddings, f, results_dict)

    _print_report(results_dict)
    _print_latex_report(results_dict)


if __name__ == '__main__':
    run()

description = 'Evaluation tool for Thai distributional models'
'with option of integrating structured sources '
'(WordNet, ConceptNet'
