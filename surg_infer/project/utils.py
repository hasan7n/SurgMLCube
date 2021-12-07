import os
import csv
import json


def _filename_no_ext(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def _file_ext(filename):
    return os.path.splitext(filename)[1]

def _get_video_fps(filename):
    cmd = f'ffmpeg -i {filename} 2>&1 | sed -n "s/.*, \(.*\) fp.*/\\1/p"'
    return round(float(os.popen(cmd).read().strip()))

def _time_str_to_sec(time_str):
    hrs, min, sec = time_str.split(":")
    hrs = int(hrs)
    min = int(min)
    sec = float(sec)
    return hrs*3600 + min*60 + sec


class LabelsParser:

    # expected to return:   list of labels or None, before processing any start, end, or fps.
    #                       i.e. a label for each single frame of the original video provided.
    

    def time_to_id(time_strs, fps):
        mapping = lambda time_str: round(fps*_time_str_to_sec(time_str))
        return list(map(mapping, time_strs))

    def check_csv_txt_structure(file):
        with open(file) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 2:
                    break
            else:
                return ","
        
        with open(file) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) != 2:
                    raise AssertionError(f"Unrecognized file structure of {file}")
            return "\t"
        

    def parse_csv_txt_labels(csv_txt_file, fps, labels_names):
        delimiter = LabelsParser.check_csv_txt_structure(csv_txt_file)
        identifiers = []
        labels = []
        with open(csv_txt_file) as f:
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                identifiers.append(row[0])
                labels.append(row[1])
        
        identifiers = identifiers[1:]
        labels = labels[1:]

        try:
            identifiers = list(map(int, identifiers))
        except ValueError:
            try:
                identifiers = LabelsParser.time_to_id(identifiers, fps)
            except ValueError:
                raise AssertionError(f"Invalid file {csv_txt_file}. Label files first column entries must be integers as frame IDs or a timestamp in the form of 'hh:mm:ss.ss'")
        

        max_len = max(identifiers)
        parsed = [None]*(max_len + 1)

        for i, frame_id in enumerate(identifiers):
            try:
                parsed[frame_id] = labels_names.index(labels[i])
            except ValueError:
                print(f"Warning: file {csv_txt_file} contains an unrecognized label: {labels[i]}")
        
        return parsed



    def parse_json_labels(json_file, fps, labels_names):
        """
        expects format from MOSAIC platform:
        a list of dict objects, each one is of at least in the form of:
            {
                'timestamp' : 
                'duration' : 
                'labelName' : 
            }
            OR
            {
                'timestamp' : 
                'duration' : 
                'label' :   {
                                'name': 
                    }
            }
        with time is in milliseconds
        """

        with open(json_file) as f:
            labels_dict = json.load(f)
        
        labels_dict.sort(key=lambda x:x['timestamp'])

        frame_id_end = 0
        parsed = []
        for phase in f:
            try:
                duration, timestamp, label = phase['duration'], phase['timestamp'], phase['labelName']
            except KeyError:
                try:
                    duration, timestamp, label = phase['duration'], phase['timestamp'], phase['label']['name']
                except KeyError:
                    raise AssertionError(f"File {json_file} structure is not supported")
            
            try:
                label_id = labels_names.index(label)
            except ValueError:
                print(f"Warning: file {json_file} contains an unrecognized label: {label}")

            frame_id_start = round(timestamp*fps/1000)

            while frame_id_end < frame_id_start:
                parsed.append(None)
                frame_id_end += 1

            frame_id_end = round((timestamp + duration)*fps/1000)

            while frame_id_start < frame_id_end:
                parsed.append(label_id)
                frame_id_start += 1
        
        return parsed

