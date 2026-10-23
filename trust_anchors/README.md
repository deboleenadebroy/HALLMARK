# Trust anchors

Put root/intermediate certificates (`.pem`, `.crt`, `.cer`, `.der`) of signing authorities you
trust here. Embedded PDF signatures that chain to one of these are reported as *trusted*;
otherwise a valid signature is reported as *valid but untrusted*.

Only add certificates you have obtained from the authority through a trustworthy channel.
This folder is git-ignored apart from this README. Revocation is not checked (offline analysis).
