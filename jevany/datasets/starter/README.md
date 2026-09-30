# Starter data

24 training records (72 questions) and 8 development records (24 questions).
These are original, synthetic support tickets with hand-authored labels, released
under the repository's Apache-2.0 license. No downloads or credentials are needed.

Each record asks for a department (`choice`), urgent review (`noul`), and priority
(`score`) under a policy included in the state. Labels follow that policy; for
example, duplicate charges require urgent review, while a request for an invoice
does not. Metadata records the source, split, license, and provenance.

Use these files to learn the format and verify a training/deployment workflow.
The development tickets are separate from training and use the same task and
policy. The released 27B checkpoint uses a different training mixture. For
application evaluation, use representative, reviewed data from your domain.
