/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#if defined(__clang__)
// clang does not support libstdc++ ranges
#include <range/v3/all.hpp>
namespace views = ranges::views;
#else
#include <ranges>
namespace views = std::views;
#endif

/// Builds a trie in parallel by splitting the input into chunks
void do_trie(std::vector<char> const &input, int domains);

int main() {
  // Books:
  char const *files[] = {"2600-0.txt", "2701-0.txt", "35-0.txt",  "84-0.txt",
                         "8800.txt",   "1727-0.txt", "55-0.txt",  "6130-0.txt",
                         "996-0.txt",  "1342-0.txt", "3825-0.txt"};

  // Read all books into a vector of characters:
  std::vector<char> input;
  for (auto *ptr : files) {
    auto const cur = input.size();

    // Find number of characters in the book:
    std::ifstream in(ptr);
    in.seekg(0, std::ios_base::end);
    auto const pos = in.tellg();

    // Resize vector of characters:
    input.resize(cur + pos);

    // Read book into vector:
    in.seekg(0, std::ios_base::beg);
    in.read((char *)input.data() + cur, pos);
  }
  std::cout << "Input size " << input.size() << " chars." << std::endl;

  // Build trie using one domain (sequentially)
  do_trie(input, 1);

  return 0;
}

/// A node of the Trie:
struct trie {
  // Pointers to children
  std::array<trie *, 26> children;
  // Number of words ending at this node
  int count = 0;
};

int index_of(char c) {
  // If character is a lower or upper case character, that's the index for the child node:
  if (c >= 'a' && c <= 'z')
    return c - 'a';
  if (c >= 'A' && c <= 'Z')
    return c - 'A';
  // All other characters are considered delimiters.
  // (do not support unicode, etc.)
  return -1;
}

void make_trie(trie &root, trie *&bump, const char *begin, const char *end, unsigned domain,
               unsigned domains);

void do_trie(std::vector<char> const &input, int domains) {
  // Allocate a vector of trie nodes
  std::vector<trie> nodes(1 << 17); // ~130k nodes
  trie *t = nodes.data();           // root of the tree
  trie **b = new trie *{t + 1};     // bump allocator for the remaining nodes

  using clk_t = std::chrono::steady_clock;
  auto const begin = clk_t::now();

  auto it = views::iota(0).begin();
  std::for_each_n(it, domains,
                  [t, b, domains, input = input.data(), size = input.size()](auto domain) {
                    make_trie(*t, *b, input, input + size, domain, domains);
                  });

  auto const time =
      std::chrono::duration_cast<std::chrono::milliseconds>(clk_t::now() - begin).count();
  auto const count = *b - nodes.data();
  std::cout << "Assembled " << count << " nodes on " << domains << " domains in " << time << "ms."
            << std::endl;
}

// Given an array of characters [`begin`, `end`), splits the array into `domains`,
// and inserts words from only one `domain` into the trie starting at `root`.
// New nodes are allocated by incrementing the `bump` allocator.
void make_trie(trie &root, trie *&bump, const char *begin, const char *end, unsigned domain,
               unsigned domains) {
  auto const size = end - begin;
  auto const domain_size = (size / domains + 1);

  // Find the boundaries of the domain:
  auto b = std::min(size, domain_size * domain);
  auto const e = std::min(size, b + domain_size);

  // Handle domains that start in the middle of a word by incrementing the domain begin such that
  // domains start at the beginning of a word:
  for (char c = begin[b]; b < size && b != e && c != 0 && index_of(c) != -1; ++b, c = begin[b])
    ;
  for (char c = begin[b]; b < size && b != e && c != 0 && index_of(c) == -1; ++b, c = begin[b])
    ;

  // Insert words of the domain into the trie: always start inserting a word at the root of the
  // trie:
  trie *n = &root;
  for (char c = begin[b];; ++b, c = begin[b]) {
    // Compute index of character into the trie node to advance to the next children
    auto const index = b >= size ? -1 : index_of(c);
    if (index == -1) {
      // If the index is a delimiter and we are inserting a word (i.e. we are not at the root node)
      // then increment the word count for the current node and go back to the root node of the trie
      if (n != &root) {
        assert(n);
        n->count += 1;
        n = &root;
      }
      // If we have completed the domain, then we are done
      if (b >= size || b > e)
        break;
      // Otherwise we continue to the next character
      else
        continue;
    }

    // The character is not a delimiter, so we need to traverse to the next node in the trie

    // If there is no child at the edge for the character we allocate it:
    if (n->children[index] == nullptr) {
      auto next = bump++;
      n->children[index] = next;
    }

    // And we traverse to it
    n = n->children[index];
  }
}
