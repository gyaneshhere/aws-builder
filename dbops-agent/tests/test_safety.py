import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"app"/"DBOpsAgent"))
from dbops.safety import validate_read_only_request
class SafetyTests(unittest.TestCase):
 def test_get_allowed(self): validate_read_only_request("GET","/_cluster/health")
 def test_post_rejected(self):
  with self.assertRaises(PermissionError): validate_read_only_request("POST","/index/_doc/1")
 def test_delete_path_rejected(self):
  with self.assertRaises(PermissionError): validate_read_only_request("GET","/index/_delete_by_query")
