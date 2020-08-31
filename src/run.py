from multiprocessing import Manager, Process

import click
import csv

from evaluation.evaluator import ThaiEvaluator
from evaluation.strategies.oov_strategies import NoActionOOVStrategy
from evaluation.strategies.base import NoStructuredSourceStrategy

from datasets.datasets import DatasetLoader, tasks
from embeddings.embeddings import load_embedding, Embedding
from cli_config import CLI_OOV_OPTION, CLI_SS_OPTION


def _print_report(results_dict):
    for task in tasks:
        print(f'Task {task}')
        print(results_dict[task])


def _print_latex_report(results_dict):
    latex = ''
    for task in tasks:
        result = results_dict[task]
        perc_oov_words = 100 * (
                result['num_missing_words'] / (result['num_found_words'] + float(result['num_missing_words'])))

        latex += '{:4.3f}~~{:4.3f}~~{:4.3f} & {:3.1f}~~{:3d}  & '.format(
            round(result['spearmanr'], 3),
            round(result['pearsonr'], 3),
            result['hm'],
            perc_oov_words,
            result['num_oov_word_pairs']
        )

    print(latex)

def _save_ranks_diff(scores_rank, y_rank, task):
    word_pairs_with_ranks = {key: abs(scores_rank.get(key, 0) - y_rank.get(key, 0))
              for key in set(scores_rank) | set(y_rank)}

    sorted_ranks = sorted(word_pairs_with_ranks.items(), key=lambda kv: kv[1])

    with open(f'{task}_ranks.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'word 1', 'word2', 'model rank', 'dataset_rank', 'abs difference'
        ])
        for sorted_rank in sorted_ranks:
            writer.writerow([
                sorted_rank[0][0],
                sorted_rank[0][1],
                scores_rank[sorted_rank[0]],
                y_rank[sorted_rank[0]],
                sorted_rank[1]
            ])


def _process_work(evaluator: ThaiEvaluator,
                  task: str,
                  embeddings: Embedding,
                  f: bool,
                  results_dict: dict):
    dataset = DatasetLoader(task)
    dataset_data = dataset.get_data()
    result = evaluator.evaluate(embeddings, dataset_data, filter_not_found=f)
    _save_ranks_diff(result['scores_rank'], result['y_rank'], task)
    results_dict[task] = result


@click.command()
@click.option("--oov", help='Strategy to handle OOV: letters or deepcut. If empty none of them is applied',
              type=click.Choice(CLI_OOV_OPTION.keys()))
@click.option("--ss", help='Integrating structed sources. If empty only word embeddings are evaluated',
              type=click.Choice(CLI_SS_OPTION.keys()))
@click.option("-f", help='Filter not found words', is_flag=True)
@click.option("-m", "--multiprocess", help='Using multiprocessing to parallel datasets evaluation', is_flag=True)
@click.argument("model")
@click.argument("format")
def run(oov, ss, f, multiprocess, model, format):
    """
    Evaluation tool for Thai distributional models with option of integrating structured sources(WordNet, ConceptNet)
    """
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
