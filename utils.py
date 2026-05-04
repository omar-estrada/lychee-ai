import csv
import os


class Csv:
    def __init__(self, filename, headers):
        exists = os.path.exists(filename)
        self._csvfile = open(filename, 'a')
        self._csvwriter = csv.writer(self._csvfile)
        if not exists:
            self._csvwriter.writerow(headers)
    
    def add_row(self, fields):
        sanitized = [self._sanitize(field) for field in fields]
        self._csvwriter.writerow(sanitized)
        self._csvfile.flush()
    
    def _sanitize(self, value):
        if type(value) is str:
            return value.replace('\n', '<NEWLINE>').replace('"', '<QUOTE>')
        else:
            return value