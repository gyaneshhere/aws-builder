import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"app"/"DBOpsAgent"))
from dbops.report import Evidence,Finding,IncidentReport
class ReportTests(unittest.TestCase):
 def test_markdown_report(self):
  r=IncidentReport("INC-1","example-domain","14:00-14:30 UTC","SEV-2",Finding("Search thread-pool saturation","High","Queue, CPU and request volume moved together.",[Evidence("OpenSearch","search queue",850,"elevated")]),["Inspect top search queries"])
  self.assertIn("Search thread-pool saturation",r.to_markdown()); self.assertIn("No production changes were performed",r.to_markdown())
