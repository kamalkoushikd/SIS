[main] INFO profile include tests: None
[main] INFO profile exclude tests: None
[main] INFO cli include tests: None
[main] INFO cli exclude tests: None
[main] INFO running on Python 3.12.11
Run started:2025-11-23 13:35:29.525684+00:00

Test results:
>> Issue: [B110:try_except_pass] Try, Except, Pass detected.
   Severity: Low Confidence: High CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.9.1/plugins/b110_try_except_pass.html
   Location: ./Digi-vuln.py:148:4
147	                print('Saved training metrics to', os.path.join(results_dir, 'training_metrics.png'))
148	            except Exception:
149	                pass
150	            else:

--------------------------------------------------

>> Issue: [B110:try_except_pass] Try, Except, Pass detected.
   Severity: Low Confidence: High CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
   More Info: https://bandit.readthedocs.io/en/1.9.1/plugins/b110_try_except_pass.html
   Location: ./Digi-vuln.py:195:8
194	                print('Saved confusion matrix to', os.path.join(results_dir, 'confusion_matrix.png'))
195	            except Exception:
196	                pass
197

--------------------------------------------------

Code scanned:
	Total lines of code: 151
	Total lines skipped (#nosec): 0

Run metrics:
	Total issues (by severity):
		Undefined: 0
		Low: 2
		Medium: 0
		High: 0
	Total issues (by confidence):
		Undefined: 0
		Low: 0
		Medium: 0
		High: 2
	Files skipped (0):