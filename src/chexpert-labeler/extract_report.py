import csv
from pathlib import Path
from ..global_paths import metadata_et_location, eyetracking_dataset_path

def load_ids(path):
    reval = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0:
                for t in row[8:23]:
                     print(t)
                continue
            id_ = row[0]
            discard = row[2].strip().lower()
            if discard == 'false':
                reval.append(id_.strip())
    return reval

def load_report(common_path, ids):
    reval = []
    for id_ in ids:
        path = common_path / id_ / 'transcription.txt'
        with open(path) as f:
            report = f.read().strip()
            if ',' in report:
                reval.append('"'+report+'"')
            else:
                reval.append(report)
    return reval


def write(reports, path):
    with open(path, 'w') as f:
        for r in reports:
            f.write(r+'\n')


if __name__ == '__main__':
    for phase in [2,3]:
        path = f'{metadata_et_location}/metadata_phase_{phase}.csv'
        ids = load_ids(path)
        common_path = Path(eyetracking_dataset_path)
        reports = load_report(common_path, ids)
        write(reports, f'phase_{phase}.csv')
