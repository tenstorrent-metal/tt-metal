// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <memory>
#include "tt_metal/common/assert.hpp"

/*
    Supports single writer, single reader
*/

template<typename T>
class LockFreeQueue {
    private:
        struct Node {
            std::shared_ptr<T> data = nullptr;
            Node* next = nullptr;
        };

        std::atomic<Node*> head;
        std::atomic<Node*> tail;

        Node* pop_head() {
            Node* oldHead = head.load();
            if (oldHead == tail.load()) {
                return nullptr; // Queue is empty
            }
            head.store(oldHead->next);
            return oldHead;
        }

    public:
        LockFreeQueue() : head(new Node), tail(head.load()) {}

        void push(const T& value) {
            std::shared_ptr<T> newData(std::make_shared<T>(value));
            Node* newNode = new Node;
            tail.load()->data = newData;
            tail.load()->next = newNode;
            tail.store(newNode);
        }

        std::shared_ptr<T> pop() {
            Node* oldHead = pop_head();
            if (!oldHead) {
                TT_THROW("Queue is empty");
            }
            std::shared_ptr<T> result(oldHead->data);
            delete oldHead;
            return result;
        }

        bool empty() const {
            return head.load() == tail.load();
        }
        class Iterator : public std::iterator<std::forward_iterator_tag, T> {
           private:
            Node* current;

           public:
            Iterator(Node* start) : current(start) {}

            Iterator& operator++() {
                if (current != nullptr) {
                    current = current->next;
                }
                return *this;
            }

            bool operator!=(const Iterator& other) const { return current != other.current; }

            const T& operator*() const { return *(current->data); }
        };

        Iterator begin() { return Iterator(head.load()); }
        Iterator end() { return Iterator(tail.load()); }
};
