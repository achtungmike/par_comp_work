import argparse
import os
import multiprocessing as mp
import mimetypes
from Bio.SeqIO.QualityIO import FastqGeneralIterator
from Bio.SeqIO.FastaIO import SimpleFastaParser
from itertools import islice
from xopen import xopen


#
# Currently only dealing with 3' adapters, which means I cut from the found
# adapter through any remaining sequence, as in the following example from
# the cutAdapt manual(http://cutadapt.readthedocs.io/en/stable/guide.html#adapters)
# Example:
# MYSEQUEN
# MYSEQUENCEADAP
# MYSEQUENCEADAPTER
# MYSEQUENCEADAPTERSOMETHINGELSE
#
# -a ADAPTER
# MYSEQUEN
# MYSEQUENCEADAP
# MYSEQUENCE
# MYSEQUENCE
#
#
# Notice, there is currently no min overlap feature, so right now we only remove
# adapters with a perfect match.  This is terrible but we need to start here.


# Argument parsing (adapter string, filetype, and file)

# Number of records to chunk off to each process.
CHUNK = 1000
END = "END"

# Parse Input
def parser():
    purpose = "parSnip removes a supplied adapter sequences from a supplied FASTA" \
              " or FASTQ file in parallel!"
    parser = argparse.ArgumentParser(description=purpose)
    parser.add_argument('-a', action="store", dest="adapter", help="Specify the adapter sequence, e.g. AGATCGGAAGAG")
    parser.add_argument('-c', action="store", dest="chunk", default=1000, help="Set the chunk size, default = 1000")
    parser.add_argument("file", action="store", help="FASTA/Q file to trim adapter from")
    args = parser.parse_args()

    # Pulling in user arguments
    adap = args.adapter
    global CHUNK
    CHUNK = int(args.chunk)
    file = args.file
    return((file,adap))

# Function to determine filetype and compression
def det_ft(file):
    # Add in our filetypes
    # FASTA (uncompressed)
    mimetypes.add_type('fasta/text', '.fasta')
    mimetypes.add_type('fasta/text', '.fa')

    # FASTA gzipped
    mimetypes.add_type('fasta/gzip', '.fasta.gz')
    mimetypes.add_type('fasta/gzip', '.fa.gz')

    # FASTQ uncompressed
    mimetypes.add_type('fastq/text', '.fastq')
    mimetypes.add_type('fastq/text', '.fq')

    # FASTQ gzipped
    mimetypes.add_type('fasta/gzip', '.fastq.gz')
    mimetypes.add_type('fasta/gzip', '.fq.gz')

    mime = mimetypes.guess_type(file)
    # Set the type
    if (mime[0] == "fasta/text"):
        ft = "fasta"
        if (mime[1] == "gzip"):
            comp = True
        else:
            comp = False
    elif (mime[0] == "fastq/text"):
        ft = "fastq"
        if (mime[1] == "gzip"):
            comp = True
        else:
            comp = False
    else:
        exit("Somehow you snuck an invalid filetype in.")
    return (ft, comp)

# Our parallel function that will do the cutting
def snip(work, adap, ft):
    d_seq_count = len(work)
    d_cut_count = 0
    ret = []

    for idx, record in enumerate(work):
        # Get the index of adapter
        index = record[1].find(adap)

        # find returns -1 if it doesn't find our adap
        if (index > -1):
            d_cut_count += 1

            # Cut the sequence string
            tmp_seq = record[1][:index]

            # Create tmp record (comment, cut_seq, cut_qual*)
            # * if fastq file'
            tmp_record = [record[0],  tmp_seq]

            # Need to deal with qual string before writing seq
            if (ft == "fastq"):
                tmp_record.append(record[2][:index])

            ret.append(tmp_record)
        # Didn't find adapter so just move record into output
        else:
            ret.append(record)
    return (ret, d_seq_count, d_cut_count)

def main():
    jobs = []
    pool = mp.Pool()


    inp = parser()
    file = inp[0]
    adap = inp[1]
    ret = det_ft(file)
    ft = ret[0]

    # Open file handle
    handle = xopen(file, 'r')

    out_file = os.path.dirname(file) + '/parSnip_' + os.path.basename(file)


    # Counters to report what work we have done
    seq_count = 0
    cut_count = 0
    with xopen(out_file, 'w') as out:
        # Process file
        if (ft == "fasta"):
            it = SimpleFastaParser(handle)
        else:
            it = FastqGeneralIterator(handle)
        while True:
            chunk = list(islice(it, CHUNK))
            if not chunk:
                break
            job = pool.apply_async(snip, (chunk, adap, ft))
            jobs.append(job)


            # Check if any jobs are done
            for job in jobs:
                # If ready pull the results, remove job from jobs list
                # Use values to update sequence count and cut count, and write the
                # trimmed or not trimmed sequence out to file with right format.
                tmp = job.get()
                jobs.remove(job)
                line = tmp[0]
                seq_count += tmp[1]
                cut_count += tmp[2]

                for l in line:
                    if (ft == "fasta"):
                        out.write(">%s\n%s\n" % (l[0], l[1]))
                    else:
                        out.write("@%s\n%s\n+%s\n%s\n" % (l[0], l[1], l[0], l[2]))

    pool.close()
    pool.join()

    print("============ Summary ============")
    print("Chunk Size: ", CHUNK)
    print("Adapter: ", adap)
    print("Input File: ", file)
    print("Output File: ", out_file)
    print("Raw Sequences: ", "{:,}".format(seq_count))
    print("Snipped Sequences: ", "{:,}".format(cut_count))
    print("Snip Percent: ", round(((cut_count / seq_count) * 100), 1))

    handle.close()


if __name__ == "__main__":
    main()