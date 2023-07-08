import json

import urllib2


class Request:
    def __init__(self, common_url):
        self.common_url = common_url

    def _request_get(self, url, data):  # pylint: disable=unused-argument
        """GET request method for flask API"""

        return

    def _request_post(self, url, data):
        """POST request method for flask API"""

        request = urllib2.Request(
            url=url,
            data=json.dumps(data),
            headers={"Content-Type": "application/json"},
        )

        try:
            response = urllib2.urlopen(request)
            response_data = response.read()

        except urllib2.HTTPError as e:
            print("HTTP Error:", e.code, e.reason)
            response_data = e.read()
            print(response_data)

        return json.loads(response_data)
