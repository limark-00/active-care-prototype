'use client';

import { useEffect, useRef } from 'react';
import type { Dispatch } from 'react';
import { requestTextDecision } from '@/lib/decision';
import {
  buildDecisionText,
  decisionPayload,
  decisionRequestKey,
} from '@/lib/care/ai';
import type { CareAction, CareEvent } from '@/lib/care/types';

/** Serializes event inference so the local model never receives a request burst. */
export function useAIEventOrchestrator(
  events: CareEvent[],
  dispatch: Dispatch<CareAction>,
  now: () => number,
  enabled = true,
) {
  const busy = useRef(false);
  const mounted = useRef(true);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (!enabled || busy.current) return;
    const event = events.find(
      (item) =>
        item.phase !== 'CLOSED' &&
        item.source !== 'manual' &&
        item.aiDecision === null,
    );
    if (!event) return;

    const requestKey = decisionRequestKey(event);
    const requestController = new AbortController();
    busy.current = true;
    controller.current = requestController;
    dispatch({ type: 'ai-start', id: event.id, requestKey, now: now() });

    void requestTextDecision(
      buildDecisionText(event.context),
      requestController.signal,
    )
      .then((result) => {
        if (!mounted.current || requestController.signal.aborted) return;
        dispatch({
          type: 'ai-result',
          id: event.id,
          requestKey,
          decision: decisionPayload(result),
          now: now(),
        });
      })
      .catch((reason: unknown) => {
        if (!mounted.current || requestController.signal.aborted) return;
        const message =
          reason instanceof Error
            ? reason.message
            : '本地AI决策服务暂时不可用。';
        dispatch({
          type: 'ai-failed',
          id: event.id,
          requestKey,
          error: message.slice(0, 240),
          now: now(),
        });
      })
      .finally(() => {
        if (controller.current === requestController) controller.current = null;
        busy.current = false;
      });
  }, [dispatch, enabled, events, now]);
}
