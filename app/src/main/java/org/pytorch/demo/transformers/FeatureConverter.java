/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package org.pytorch.demo.transformers;

import org.pytorch.demo.tokenization.FullTokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Convert String to features that can be fed into BERT model.
 */
public final class FeatureConverter {
    private final FullTokenizer tokenizer;
    private final int maxSeqLen;
    private final boolean padToMaxLength;

    public FeatureConverter(
            Map<String, Integer> inputDic, boolean doLowerCase, int maxSeqLen, boolean padToMaxLength) {
        this.tokenizer = new FullTokenizer(inputDic, doLowerCase);
        this.maxSeqLen = maxSeqLen;
        this.padToMaxLength = padToMaxLength;
    }

    public Feature convert(String text, boolean addSpecialTokens) {
        List<String> tokens = tokenizer.tokenize(text);
        if (addSpecialTokens) {
            tokens = tokenizer.addSpecialToken(tokens, maxSeqLen);
        }

        List<Integer> inputIds = tokenizer.convertTokensToIds(tokens);
        if (padToMaxLength) {
            while (inputIds.size() < maxSeqLen) {
                inputIds.add(0);
            }
        }
        return new Feature(inputIds);
    }


    public Feature convert(String text) {
        text = convertWords(text);

        String[] words = text.split(" ");
        List<String> tokens = new ArrayList<>();
        for (String word : words) {
            if(!word.equals("")){
                tokens.add(word);
            }
        }
        if (tokens.size() > maxSeqLen - 1) {
            tokens = tokens.subList(0, maxSeqLen - 1);
        }

        List<Integer> inputIds = tokenizer.convertTokensToIds(tokens);
        if (padToMaxLength) {
            while (inputIds.size() < maxSeqLen) {
                inputIds.add(0);
            }
        }
        return new Feature(inputIds);
    }

    String convertWords(String sentence) {
        List<String> newWords = new ArrayList<>();
        Pattern pattern = Pattern.compile("(?i)\\b\\w\\w+\\b");
        Matcher matcher = pattern.matcher(sentence.toLowerCase());
        while (matcher.find()) {
            String word = matcher.group();
            newWords.add(word);
        }
        return String.join(" ", newWords);
    }
}
