package com.np_solr;

import org.apache.lucene.analysis.payloads.PayloadHelper;
import org.apache.lucene.index.*;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.DoubleValues;
import org.apache.lucene.search.DoubleValuesSource;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.*;

/*
 * This class receives the query vector and computes its distance to the document vector by reading the vector values directly from the Lucene index. As distance metric, the Jensen-Shannon divergence is used.
 */
public class VectorSimValuesSource extends DoubleValuesSource {
    private final String field;
    private final String distance;

    private Terms terms; // Access to the terms in a specific field
    private TermsEnum te; // Iterator to step through terms to obtain frequency information
    private String[] query_comps;
    private boolean float_flag;

    public VectorSimValuesSource(String field, String strVector, String distance) {
        /*
         * Document queries are assumed to be given as:
         * http://localhost:8983/solr/{your-corpus-collection-name}/query?fl=name,score,
         * vector&q={!vp f=doctpc_{your-model-name} vector="t0|43 t4|548 t5|6 t20|403"}
         * while topic queries as follows:
         * http://localhost:8983/solr/{your-model-collection-name}/query?fl=name,score,
         * vector&q={!vp f=betas
         * vector="high|43 research|548 development|6 neural_networks|403"}
         */
        this.field = field;
        this.query_comps = strVector.split(" ");
        this.distance = distance;
        this.float_flag = false;
    }

    public DoubleValues getValues(LeafReaderContext leafReaderContext, DoubleValues doubleValues) throws IOException {

        final LeafReader reader = leafReaderContext.reader();

        return new DoubleValues() {

            // Retrieves the payload value for each term in the document and calculates the
            // core based on vector lookup
            @SuppressWarnings("unchecked")
            public double doubleValue() throws IOException {
                double score = 0;
                BytesRef text;
                String term = "";
                List<String> doc_topics = new ArrayList<String>();
                List<? extends Number> doc_probs;
                if (float_flag == true) {
                    doc_probs = new ArrayList<Float>();
                } else{
                    doc_probs = new ArrayList<Integer>();
                }
                while ((text = te.next()) != null) {
                    term = text.utf8ToString();
                    if (term.isEmpty()) {
                        continue;
                    }

                    // Get the document probability distribution
                    float payloadValue = 0f;
                    PostingsEnum postings = te.postings(null, PostingsEnum.ALL);
                    // And after we get TermsEnum instance te, we can compute the document vector by
                    // iterating all payload components (we will have as many components as topics
                    // the model has)
                    while (postings.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                        int freq = postings.freq();
                        while (freq-- > 0)
                            postings.nextPosition();

                        BytesRef payload = postings.getPayload();
                        if (float_flag == true) {
                            payloadValue = PayloadHelper.decodeFloat(payload.bytes, payload.offset);
                            ((List<Float>) doc_probs).add(payloadValue);
                        } else {
                            payloadValue = PayloadHelper.decodeInt(payload.bytes, payload.offset);
                            ((List<Integer>) doc_probs).add((int) payloadValue);
                        }

                        doc_topics.add(term);
                    }
                }

                // Create maps containing the value after '|' for each t that is present in both
                // strings for the case of document queries, and for each word that is present
                // in both strings for the case of topic queries
                //Map<String, Integer> doc_values = new HashMap<>();
                //Map<String, Integer> query_values = new HashMap<>();

                Map<String, Number> doc_values = new HashMap<>();
                Map<String, Number> query_values = new HashMap<>();

                // Create pattern to match the document, topic, and embedding queries
                Pattern pattern_docs = Pattern.compile("(t\\d+)\\|");
                Pattern pattern_words = Pattern.compile("([^|]+)\\|(\\d+)");
                Pattern pattern_embs = Pattern.compile("(e\\d+)\\|");

                // int aux = 0;
                // if (aux == 0) {
                //     throw new IllegalArgumentException(Arrays.toString(query_comps));
                // }

                for (String comp : query_comps) {
                    String key = "";
                    Matcher matcher;

                    matcher = pattern_docs.matcher(comp);
                    if (matcher.find()) {
                        key = matcher.group(1);
                    } else {
                        matcher = pattern_embs.matcher(comp);
                        if (matcher.find()) {
                            key = matcher.group(1);
                            float_flag = true;
                        }
                        else {
                            matcher = pattern_words.matcher(comp);
                            if (matcher.find()) {
                                key = matcher.group(1);
                            }
                        }
                    }

                    if (doc_topics.contains(key)) {
                        if (float_flag == true)
                            query_values.put(key, Float.parseFloat(comp.split("\\|")[1]));
                        else
                            query_values.put(key, Integer.parseInt(comp.split("\\|")[1]));
                        doc_values.put(key, doc_probs.get(doc_topics.indexOf(key)));
                    }
                }

                // Convert the maps into arrays
                List<String> keys = new ArrayList<>(doc_values.keySet());

                double[] docProbabilities = new double[keys.size()];
                double[] queryProbabilities = new double[keys.size()];

                for (int i = 0; i < keys.size(); i++) {
                    String t = keys.get(i);
                    docProbabilities[i] = doc_values.get(t).doubleValue();
                    queryProbabilities[i] = query_values.get(t).doubleValue();
                }

                System.out.println(Arrays.toString(docProbabilities));
                System.out.println(Arrays.toString(queryProbabilities));

                Distance d = new Distance();

                if (VectorSimValuesSource.this.distance.equals("bhattacharyya")) {
                    System.out.println("Using Bhattacharyya distance");
                    score = d.bhattacharyyaDistance(docProbabilities, queryProbabilities);
                } else if (VectorSimValuesSource.this.distance.equals("kullback-leibler")) {
                    System.out.println("Using Kullback-Leibler divergence");
                    score = d.KullbackLeiblerDivergence(docProbabilities, queryProbabilities);
                } else if (VectorSimValuesSource.this.distance.equals("cosine")) {
                    System.out.println("Using Cosine similarity");
                    score = d.cosineDistance(docProbabilities, queryProbabilities);
                } else {
                    System.out.println("Using default metric: Jensen-Shannon divergence");
                    score = d.JensenShannonDivergence(docProbabilities, queryProbabilities);
                }

                return score;

            }

            // Advance to next document (for each document in the LeafReaderContext)
            public boolean advanceExact(int doc) throws IOException {
                terms = reader.getTermVector(doc, field);
                if (terms == null) {
                    return false;
                }
                te = terms.iterator();
                return true;
            }
        };
    }

    public boolean needsScores() {
        return true;
    }

    public DoubleValuesSource rewrite(IndexSearcher indexSearcher) throws IOException {
        return this;
    }

    public int hashCode() {
        return 0;
    }

    public boolean equals(Object o) {
        return false;
    }

    public String toString() {
        return "JS(" + field + ",doc)";
    }

    public boolean isCacheable(LeafReaderContext leafReaderContext) {
        return false;
    }
}