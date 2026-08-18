import argparse
import os
import time

import pandas as pd

from src.surgery_pipeline import SurgeryPipeline
from src.clinic_pipeline import ClinicPipeline

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SURGERY_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'surgery')
CLINIC_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'clinic')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'reports')
FIGURE_DIR = os.path.join(REPORT_DIR, 'figures')

SURGERY_DEPARTMENTS = ['ENT', 'GS', 'GU', 'OPH', 'ORTH']

CLINIC_DEPARTMENTS = {
    'cardiology': {'department': 'Cardiology', 'wait_min': -100, 'diag_max': 80},
    'neurology': {'department': 'Neurology', 'wait_max': 200, 'diag_max': 50},
    'rheumatology': {'department': 'Rheumatology', 'wait_min': -200, 'diag_max': 25},
}


def run_surgery(dept: str) -> list:
    print(f'\n================ Surgery: {dept} ================')
    pipeline = SurgeryPipeline(dept, SURGERY_DATA_DIR, OUTPUT_DIR, FIGURE_DIR)
    pipeline.run()
    return pipeline.metrics


def run_clinic(key: str) -> list:
    config = CLINIC_DEPARTMENTS[key]
    print(f'\n================ Clinic: {config["department"]} ================')
    pipeline = ClinicPipeline(
        department=config['department'],
        filename=f'{key}.xlsx',
        data_dir=CLINIC_DATA_DIR,
        output_dir=OUTPUT_DIR,
        figure_dir=FIGURE_DIR,
        wait_min=config.get('wait_min'),
        wait_max=config.get('wait_max'),
        diag_max=config.get('diag_max', 100.0),
    )
    pipeline.run()
    return pipeline.metrics


def save_metrics(rows: list, filename: str) -> None:
    if not rows:
        return
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, filename)
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        # 單科重跑時保留其他科別既有結果
        old = pd.read_csv(path, encoding='utf-8-sig')
        old = old[~old['department'].isin(df['department'].unique())]
        df = pd.concat([old, df], ignore_index=True)
    df = df.sort_values(['department', 'model'])
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'指標已彙整: {path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='醫療時間預測：手術時間迴歸與門診延誤分類')
    parser.add_argument('task', choices=['surgery', 'clinic', 'all'], help='要執行的任務')
    parser.add_argument(
        'department', nargs='?', default=None,
        help='科別（surgery: ENT/GS/GU/OPH/ORTH；clinic: cardiology/neurology/rheumatology；省略則跑全部）'
    )
    args = parser.parse_args()

    surgery_depts, clinic_keys = [], []
    if args.task in ('surgery', 'all'):
        if args.task == 'surgery' and args.department:
            dept = args.department.upper()
            if dept not in SURGERY_DEPARTMENTS:
                parser.error(f'未知的手術科別: {args.department}（可用: {", ".join(SURGERY_DEPARTMENTS)}）')
            surgery_depts = [dept]
        else:
            surgery_depts = SURGERY_DEPARTMENTS
    if args.task in ('clinic', 'all'):
        if args.task == 'clinic' and args.department:
            key = args.department.lower()
            if key not in CLINIC_DEPARTMENTS:
                parser.error(f'未知的門診科別: {args.department}（可用: {", ".join(CLINIC_DEPARTMENTS)}）')
            clinic_keys = [key]
        else:
            clinic_keys = list(CLINIC_DEPARTMENTS)

    start = time.time()
    surgery_rows, clinic_rows = [], []
    for dept in surgery_depts:
        surgery_rows += run_surgery(dept)
    for key in clinic_keys:
        clinic_rows += run_clinic(key)

    save_metrics(surgery_rows, 'metrics_surgery.csv')
    save_metrics(clinic_rows, 'metrics_clinic.csv')
    print(f'\n總耗時: {time.time() - start:.1f} 秒')


if __name__ == '__main__':
    main()
