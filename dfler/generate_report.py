import pdfkit
import socket
import json
from datetime import datetime
import os
import pandas as pd
# import dfler from output_dir


def build_head(report_html):
  report = open(report_html, 'a')
  report.write("""
  <!DOCTYPE html>
  <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta http-equiv="X-UA-Compatible" content="IE=edge" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>Forensic Report</title>
    </head>
  """)
  report.close()


def build_foot(config, report_html):
  report = open(report_html, 'a')
  report.write("""
      </body>
    <footer id="footer">
      <span class="timestamp"> <em>Generated using dfler {app_version}</em> </span>
    </footer>
  </html>
  """.format(app_version=config['app_version']))
  report.close()


def build_style(report_html):
  report = open(report_html, 'a')
  report.write("""
  <style>
    body {
      font-family: Arial, Helvetica, sans-serif;
    }
    .text-center {
      text-align: center;
    }
    th {
      text-align: center;
    }

    th,
    td {
      padding-top: 5px;
      padding-bottom: 5px;
      padding-left: 8px;
      padding-right: 8px;
    }

    /* table,
    th,
    td {
      border: 1px solid #000;
    } */

    .timeline {
      font-family: Arial, Helvetica, sans-serif;
      font-size: 1.2em;
      text-align: center;
      font-weight: bold;
    }

    table.fixed {
      table-layout: fixed;
    }
    table.fixed td {
      overflow: hidden;
    }

    table {
      border-collapse: collapse;
      font-family: Arial, Helvetica, sans-serif;
      font-size: 0.95em;
    }

    table.table-timeline {
      width: 100%;
      font-size: 1em;
    }

    table.table-timeline > thead { display: table-header-group; }
    table.table-timeline > tfoot { display: table-row-group; }
    table.table-timeline > tbody > tr { page-break-inside: avoid; }

    table.table-timeline,
    table.table-timeline > thead > tr > th,
    table.table-timeline > tbody > tr > td {
      border: 1px solid #000;
    }

    table.table-timeline > thead > tr > th {
      font-weight: bold;
    }

    table.table-timeline > tbody > tr > td:first-child {
      text-align: center;
    }

    .outside {
      color: #000;
    }

    .issue {
      color: red;
    }

    .parameter {
      color: coral;
    }

    .action {
      color: rgb(185, 0, 0);
    }

    .component {
      color: blue;
    }

    .function {
      color: blueviolet;
    }

    .state {
      color: #009eff;
    }

    .bold {
      text-emphasis-color: red;
    }

    #footer {
      margin-top: 7px;
      display: flex;
      justify-content: space-between;
    }

    .content-color {
      color: #2b4f60;
    }

    .timestamp {
      color: grey;
      font-size: small;
    }
    .page-number {
      width: 50%;
      text-align: right;
      margin-right: 5px;
    }

    .report-title {
      font-weight: bold;
      text-align: left;
      font-size: 1.8em;
      color: #0e5e6f;
    }

    .title-color {
      color: #0e5e6f;
      font-size: 1em;
    }

    #metadata {
      margin-top: 3em;
    }

    #table-metadata {
      border: none !important;
      width: 30%;
      font-size: 1em;
      margin-left: -8px;
    }

    #color-legend.table-timeline {
      width: 9%;
      border: none !important;
    }

    .box {
      float: left;
      width: 20px;
      height: 20px;
      margin: 5px;
      border: 1px solid rgba(0, 0, 0, 0.2);
    }

    .outside-box {
      background: #000;
    }

    .issue-box {
      background: red;
    }

    .parameter-box {
      background: coral;
    }

    .action-box {
      background: rgb(185, 0, 0);
    }

    .component-box {
      background: blue;
    }

    .function-box {
      background: blueviolet;
    }

    .state-box {
      background: #009eff;
    }

    .break-before {
      page-break-before: always;
    }
  </style>
  """)
  report.close()


def build_report_header(config, report_html):
  now = datetime.now()
  now = now.strftime("%m/%d/%Y %H:%M:%S")
  hostname = socket.gethostname()
  raw_list = open(config['output_dir'] + '/raw_list.json')
  raw_list = json.load(raw_list)
  flat_list = [item for sublist in raw_list for item in sublist]
  report = open(report_html, 'a')
  report.write("""
  <body>
    <h4 class="report-title">Drone Forensic Report</h4>
    <hr style="margin-top: -2em" />
    <span class="timestamp"
      >This report is generated on: {timestamp}</span
    >
    <section id="metadata">
      <table id="table-metadata fixed">
        <col width="30%" />
        <col width="70%" />
        <tr>
          <td>Computer Name</td>
          <td>{hostname}</td>
        </tr>
        <tr>
          <td>Report Type</td>
          <td>Entity Recognition</td>
        </tr>
        <tr>
          <td>Number of log files</td>
          <td>{num_evidence}</td>
        </tr>
      </table>
    </section>
  """.format(timestamp=now, hostname=hostname, num_evidence=len(flat_list)))
  report.close()


def build_source_evidence(config, report_html):
  content = """
    <section>
      <h4 class="title-color" style="margin-top: 3em">Source evidence</h4>
      <hr style="margin-top: -1em" />
      <ul class="content-color">
  """
  raw_list = open(config['output_dir'] + '/raw_list.json')
  raw_list = json.load(raw_list)
  flat_list = [item for sublist in raw_list for item in sublist]

  for item in flat_list:
    content = content + """
        <li>{filename}</li>
    """.format(filename=item) 
    # print("<li>{filename}</li>".format(filename=item))
  content = content + """
      </ul>
    </section>
  """
  report = open(report_html, 'a')
  report.write(content)
  report.close()


def build_ner_result(statistics, report_html):
  report = open(report_html, 'a')
  content = """
    <section>
      <h4 class="title-color" style="margin-top: 3em">Recognition Results</h4>
      <hr style="margin-top: -1em" />
      <table id="table-metadata fixed">
        <col width="40%" />
        <col width="10%" />
        <col width="40%" />
        <col width="10%" />
    """
  counter = 1
  for key, value in statistics.items():
    if (counter % 2 == 1):
      content = content + """
        <tr>
          <td>Number of {key}</td>
          <td>{value}</td>
      """.format(key=key, value=value)
    else:
      content = content + """
          <td>Number of {key}</td>
          <td>{value}</td>
        </tr>
      """.format(key=key, value=value)
    counter = counter + 1
  content = content + """
      </table>
    </section>
  """
  report.write(content)
  report.close()

def build_th(report_html):
  report = open(report_html, 'a')
  report.write("""
  <section class="break-before">
    <h5 class="timeline">Highlights Color Code</h5>
      <table class="table-timeline fixed">
        <col width="10%" />
        <col width="20%" />
        <col width="70%" />
        <thead>
          <tr>
            <th>Color Code</th>
            <th>Entity Type</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <span class="box issue-box"></span>
            </td>
            <td>
              <span class="issue">Issue</span>
            </td>
            <td>
              Words/phrases that indicate some issues happen to the drone.
            </td>
          </tr>
          <tr>
            <td>
              <span class="box parameter-box"></span>
            </td>
            <td>
              <span class="parameter">Parameter</span>
            </td>
            <td>
              Words/phrases that represent some parameters of configuration in a
              drone.
            </td>
          </tr>
          <tr>
            <td>
              <span class="box action-box"></span>
            </td>
            <td>
              <span class="action">Action</span>
            </td>
            <td>
              Words/phrases that indicate some actions taken by the drone.
            </td>
          </tr>
          <tr>
            <td>
              <span class="box component-box"></span>
            </td>
            <td><span class="component">Component</span></td>
            <td>Words/phrases that reflect physical components of a drone</td>
          </tr>
          <tr>
            <td>
              <span class="box function-box"></span>
            </td>
            <td><span class="function">Function</span></td>
            <td>
              Words/phrases that denote some functionalities or features of a
              drone equipped with.
            </td>
          </tr>
          <tr>
            <td>
              <span class="box state-box"></span>
            </td>
            <td><span class="state">State</span></td>
            <td>
              Words/phrases that notify a state/mode of a drone operates in
              during a flight.
            </td>
          </tr>
        </tbody>
      </table>
      <h5 class="timeline" style="margin-top: 3em">
        Highlighted Forensic Timeline
      </h5>
      <table class="table-timeline fixed">
        <col width="30%" />
        <col width="70%" />
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Message</th>
          </tr>
        </thead>
        <tbody>
      """)
  report.close()

def statistics(config, ner_result):
  entities_json = open('./flight_logs/entities.json')
  entities = json.load(entities_json)

  word_list = []
  tag_list = []
  for record in entities:
      timestamp = record['timestamp']
      messages = record['entities']
      for message in messages:
          for word, tag in message.items():
              word_list.append(word)
              tag_list.append(tag)

  ner_result = pd.DataFrame(list(zip(word_list, tag_list)), columns =['word', 'tag'])
  entity_df = ner_result[ner_result['tag'] != 'O']
  non_entity_df = ner_result[ner_result['tag'] == 'O']
  issue = ['B-ISSUE', 'I-ISSUE']
  issue_df = ner_result[ner_result['tag'].isin(issue)]
  component = ['B-COMPONENT', 'I-COMPONENT']
  component_df = ner_result[ner_result['tag'].isin(component)]
  action = ['B-ACTION', 'I-ACTION']
  action_df = ner_result[ner_result['tag'].isin(action)]
  parameter = ['B-PARAMETER', 'I-PARAMETER']
  parameter_df = ner_result[ner_result['tag'].isin(parameter)]
  state = ['B-STATE', 'I-STATE']
  state_df = ner_result[ner_result['tag'].isin(state)]
  function = ['B-FUNCTION', 'I-FUNCTION']
  function_df = ner_result[ner_result['tag'].isin(function)]
  entities_json.close()
  return {
    'message': len(entities),
    'entity': len(entity_df),
    'non_entity': len(non_entity_df),
    'issue': len(issue_df),
    'component': len(component_df),
    'action': len(action_df),
    'parameter': len(parameter_df),
    'state': len(state_df),
    'function': len(function_df),
  }


def build_forensic_table(config, report_html):
  # Opening JSON file
  timeline_file = open(config['output_dir'] + '/ner_result.json')
  timeline = json.load(timeline_file)
  
  build_th(report_html)
  # Loop the table
  build_tr(timeline, report_html)
  # Closing file
  timeline_file.close()


def build_tr(records, report_html):
  report = open(report_html, 'a')
  content = ""
  for record in records:
    timestamp = record['timestamp']
    content = content + """
          <tr>
            <td>{timestamp}</td>
            <td>
    """.format(timestamp=timestamp)
    messages = record['entities']
    for message in messages:
      for word, tag in message.items():
        if tag == 'O':
          # generate tag span O
          content = content + """
              <span class="{tag}">{token}</span>
          """.format(tag='outside', token=word)
        elif tag == 'B-ISSUE' or tag == 'I-ISSUE':
          content = content + """
              <span class="{tag}">{token}</span>
          """.format(tag='issue', token=word)
        elif tag == 'B-PARAMETER' or tag == 'I-PARAMETER':
          content = content + """
              <span class="{tag}">{token}</span>
          """.format(tag='parameter', token=word)
        elif tag == 'B-ACTION' or tag == 'I-ACTION':
          content = content + """
              <span class="{tag}">{token}</span>
          """.format(tag='action', token=word)
        elif tag == 'B-COMPONENT' or tag == 'I-COMPONENT':
          content = content + """
              <span class="{tag}">{token}</span>
          """.format(tag='component', token=word)
        elif tag == 'B-FUNCTION' or tag == 'I-FUNCTION':
          content = content + """
              <span class="{tag}">{token}</span>
          """.format(tag='function', token=word)
        elif tag == 'B-STATE' or tag == 'I-STATE':
          content = content + """
              <span class="{tag}">{token}</span>
          """.format(tag='state', token=word)

    content = content + """
            </td>
          </tr>
    """  
  content = content + """
        </tbody>
      </table>
    </section>
  """
  report.write(content)
  report.close()

def statistical_analysis(config):
  # Opening JSON file
  ner_result_json = open(config['output_dir'] + '/ner_result.json')
  ner_result = json.load(ner_result_json)

  word_list = []
  tag_list = []
  for record in ner_result:
      timestamp = record['timestamp']
      messages = record['entities']
      for message in messages:
          for word, tag in message.items():
              word_list.append(word)
              tag_list.append(tag)
  
  ner_result_df = pd.DataFrame(list(zip(word_list, tag_list)), columns =['word', 'tag'])
  entity_df = ner_result_df[ner_result_df['tag'] != 'O']
  non_entity_df = ner_result_df[ner_result_df['tag'] == 'O']
  issue = ['B-ISSUE', 'I-ISSUE']
  issue_df = ner_result_df[ner_result_df['tag'].isin(issue)]
  component = ['B-COMPONENT', 'I-COMPONENT']
  component_df = ner_result_df[ner_result_df['tag'].isin(component)]
  action = ['B-ACTION', 'I-ACTION']
  action_df = ner_result_df[ner_result_df['tag'].isin(action)]
  parameter = ['B-PARAMETER', 'I-PARAMETER']
  parameter_df = ner_result_df[ner_result_df['tag'].isin(parameter)]
  state = ['B-STATE', 'I-STATE']
  state_df = ner_result_df[ner_result_df['tag'].isin(state)]
  function = ['B-FUNCTION', 'I-FUNCTION']
  function_df = ner_result_df[ner_result_df['tag'].isin(function)]
  statistics = {
    "message": len(ner_result),
    "entity": len(entity_df),
    "non_entity": len(non_entity_df),
    "token": len(entity_df) + len(non_entity_df),
    "state": len(state_df),
    "issue": len(issue_df),
    "action": len(action_df),
    "function": len(function_df),
    "component": len(component_df),
    "parameter": len(parameter_df)
  }

  with open(config['output_dir'] + '/statistics.json', 'w') as file:
    json.dump(statistics, file)
  ner_result_json.close()
  return statistics

def build_html(config, filename):
  output_dir = config['output_dir']
  full_path = os.path.join(output_dir, filename + ".html")
  statistics = statistical_analysis(config)

  # sys.stdout = open(full_path, 'w')
  build_head(full_path)
  build_style(full_path)
  build_report_header(config, full_path)
  build_source_evidence(config, full_path)
  build_ner_result(statistics, full_path)
  build_forensic_table(config, full_path)
  build_foot(config, full_path)
  # sys.stdout.close()


def generatePDF(config, filename):
  # Define path to wkhtmltopdf.exe
  # path_to_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
  path_to_wkhtmltopdf = config['wkhtml_path']
  # Define path to input and output file
  output_dir = config['output_dir']
  full_path = os.path.join(output_dir, filename)
  # Point pdfkit configuration to wkhtmltopdf.exe
  config_wkhtml = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)
  # Convert HTML file to PDF
  pdfkit.from_file(full_path + ".html", output_path = full_path + ".pdf", configuration=config_wkhtml)


def generate_report(config):
  # Prepare the filename and outputdir
  # Move to the config, so that every function can access
  # now = datetime.now()
  # now = now.strftime("%d%m%Y_%H%M%S")
  # output_dir = os.path.join("./result", now)
  filename = "forensic_report_"

  build_html(config, filename)
  generatePDF(config, filename)
  
    