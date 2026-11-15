import unittest

from pipeline.list_cloud_spectral_wcs import list_spectral_wcs, parse_list_response


S3 = "http://s3.amazonaws.com/doc/2006-03-01/"


class CloudSpectralWcsTests(unittest.TestCase):
    def test_parses_object_metadata_and_pagination(self):
        xml = f'''<ListBucketResult xmlns="{S3}">
          <IsTruncated>true</IsTruncated>
          <NextContinuationToken>next-page</NextContinuationToken>
          <Contents><Key>dr1/spectral_wcs/cal-wcs/D1/map.fits</Key>
            <LastModified>2026-11-01T00:00:00.000Z</LastModified><ETag>"abc123"</ETag><Size>42</Size>
          </Contents>
        </ListBucketResult>'''
        parsed = parse_list_response(xml)
        self.assertTrue(parsed["is_truncated"])
        self.assertEqual(parsed["next_token"], "next-page")
        self.assertEqual(parsed["objects"], [{
            "key": "dr1/spectral_wcs/cal-wcs/D1/map.fits", "size_bytes": 42,
            "last_modified": "2026-11-01T00:00:00.000Z", "etag": "abc123",
        }])

    def test_lists_all_pages_and_filters_to_fits(self):
        class Response:
            status_code = 200

            def __init__(self, text):
                self.text = text

        class Session:
            def __init__(self):
                self.calls = []

            def get(self, url, params, timeout):
                self.calls.append((url, params, timeout))
                if len(self.calls) == 1:
                    return Response(f'''<ListBucketResult xmlns="{S3}">
                      <IsTruncated>true</IsTruncated><NextContinuationToken>page2</NextContinuationToken>
                      <Contents><Key>qr3/spectral_wcs/a.fits</Key><Size>10</Size></Contents>
                      <Contents><Key>qr3/spectral_wcs/readme.txt</Key><Size>2</Size></Contents>
                    </ListBucketResult>''')
                return Response(f'''<ListBucketResult xmlns="{S3}">
                  <IsTruncated>false</IsTruncated>
                  <Contents><Key>qr3/spectral_wcs/b.fits</Key><Size>20</Size></Contents>
                </ListBucketResult>''')

        session = Session()
        result = list_spectral_wcs("qr3", session=session)
        self.assertEqual([item["key"] for item in result], [
            "qr3/spectral_wcs/a.fits", "qr3/spectral_wcs/b.fits",
        ])
        self.assertEqual(session.calls[1][1]["continuation-token"], "page2")

    def test_rejects_unknown_release(self):
        with self.assertRaises(ValueError):
            list_spectral_wcs("spherex_dr1")


if __name__ == "__main__":
    unittest.main()
