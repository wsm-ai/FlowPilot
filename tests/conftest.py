import os


# Historical tests exercise business behavior independently of the HTTP auth
# boundary. Authentication is disabled explicitly for pytest only; production
# defaults remain fail-closed and dedicated auth tests enable it themselves.
os.environ["FLOWPILOT_AUTH_ENABLED"] = "false"
