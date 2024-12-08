import pandas as pd
import editdistance as ed
import csv
import sys
from tqdm import tqdm

def calculate_metrics(predicted_text, transcript):
    cer = ed.eval(predicted_text, transcript) / max(len(predicted_text), len(transcript))
    pred_spl = predicted_text.split()
    transcript_spl = transcript.split()
    wer = ed.eval(pred_spl, transcript_spl) /  max(len(pred_spl), len(transcript_spl))
    return cer, wer

def get_metric(filename, output_filename):
    cer_list = []
    wer_list = []
    df_output = pd.read_csv(filename)
        
    with open(filename, "r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  
        
        for row in tqdm(csv_reader,desc="Processing"):
            ref = row[1].strip()
            output = row[0].strip()
            
            cer, wer = calculate_metrics(output, ref)
            cer_list.append(cer)
            wer_list.append(wer)
                
    df_output['cer_c_o'] = cer_list
    df_output['wer_c_o'] = wer_list
    mean_cer = df_output['cer_c_o'].mean()
    mean_wer = df_output['wer_c_o'].mean()
    print(f'Mean CER = {mean_cer}, Mean WER = {mean_wer}')
    
    # Save the rows with metrics to a new CSV file
    # with open(output_filename, "w", encoding="utf-8", newline="") as out_file:
    #     csv_writer = csv.writer(out_file)
    #     csv_writer.writerows(rows_with_metrics)


if __name__ == "__main__":
    input_filename = sys.argv[1]
    output_filename = input_filename.split('.')[0].split('/')[-1] + '_cer_wer.csv'
    print('Processing started')
    cer, wer = get_metric(input_filename, output_filename)
